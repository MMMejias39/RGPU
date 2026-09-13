//! Backend de GPU via `wgpu` (Vulkan/Metal/DX12/WebGPU), ativado pela feature `gpu`.
//!
//! # Desenho
//!
//! Os pesos ficam **residentes na GPU** do início ao fim do treino: não há
//! cópia host↔device dentro do laço. Um passo inteiro — avanço, retropropagação
//! e atualização do Adam — é gravado num único `CommandEncoder` e submetido de
//! uma vez, sem sincronização. A perda só é baixada quando pedida.
//!
//! # Operações pré-ligadas
//!
//! Criar um buffer uniforme e um bind group por dispatch custa caro no lado da
//! CPU, e esse custo domina redes profundas — muitas camadas pequenas gastam
//! mais tempo montando descritores do que calculando. Aqui uma [`Op`] carrega
//! pipeline, bind group, uniforme e grade já resolvidos: [`GpuMlp`] monta o
//! plano inteiro **uma vez**, na construção, e cada passo apenas o regrava.
//!
//! Só os uniformes do Adam mudam entre passos (a correção de viés depende de
//! `t`), e são reescritos com `write_buffer` sem refazer o bind group.
//!
//! # Perfilamento
//!
//! Quando o adaptador oferece `TIMESTAMP_QUERY`, cada passe de computação é
//! cercado por um par de marcas de tempo. [`Gpu::perfilar`] devolve o tempo de
//! cada kernel medido **no dispositivo** — sem chute e sem interferência do
//! relógio do host.
//!
//! # Relação com o motor de CPU
//!
//! O caminho de CPU deriva os gradientes pela fita ([`crate::tape`]); aqui a
//! retropropagação da MLP é escrita à mão, exatamente como o TensorFlow faz nos
//! seus kernels fundidos. As fórmulas são as mesmas registradas em
//! [`crate::nn`] — e `tests/gpu.rs` confere numericamente uma contra a outra.
//!
//! ```text
//! Z¹ = X W¹ + 1ₙb¹      A¹ = relu(Z¹)
//! Z² = A¹W² + 1ₙb²      A² = relu(Z²)
//! Z³ = A²W³ + 1ₙb³      L  = xent(softmax(Z³), y)
//!
//! δ³ = (P − Y)/n
//! ∂L/∂W³ = (A²)ᵀδ³   ∂L/∂b³ = 1ₙᵀδ³   Ā² = δ³(W³)ᵀ
//! δ² = Ā² ⊙ 1[Z² > 0]
//! ∂L/∂W² = (A¹)ᵀδ²   ∂L/∂b² = 1ₙᵀδ²   Ā¹ = δ²(W²)ᵀ
//! δ¹ = Ā¹ ⊙ 1[Z¹ > 0]
//! ∂L/∂W¹ = Xᵀδ¹      ∂L/∂b¹ = 1ₙᵀδ¹
//! ```

mod gemm;
mod kernels;

use std::cell::{Cell, RefCell};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::tape::Param;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct P4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamP {
    n: u32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    c1: f32,
    c2: f32,
    gx: u32,
}

/// Identifica um pipeline compilado. Guardar a variante em vez do objeto evita
/// clonar handles e mantém a [`Op`] barata de construir e de copiar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Mm,
    MmAtb,
    MmAbt,
    MmF,
    MmAtbF,
    MmAbtF,
    BiasAdd,
    Relu,
    ReluBwd,
    Xent,
    Colsum,
    Adam,
}

/// Uma operação com tudo já resolvido: só falta gravá-la num encoder.
pub struct Op {
    kernel: Kernel,
    bind: wgpu::BindGroup,
    uniforme: wgpu::Buffer,
    gx: u32,
    gy: u32,
    rotulo: &'static str,
}

/// Tensor residente na memória da GPU.
pub struct GpuTensor {
    shape: Vec<usize>,
    len: usize,
    buf: wgpu::Buffer,
}

impl GpuTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Marcas de tempo do dispositivo.
struct Perfil {
    conjunto: wgpu::QuerySet,
    resolucao: wgpu::Buffer,
    leitura: wgpu::Buffer,
    capacidade: u32,
    /// Nanossegundos por tique do contador da GPU.
    periodo: f32,
    proximo: Cell<u32>,
    rotulos: RefCell<Vec<&'static str>>,
    ativo: Cell<bool>,
}

/// Contexto de GPU: dispositivo, fila e os pipelines de computação compilados.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: String,
    mm: wgpu::ComputePipeline,
    mm_atb: wgpu::ComputePipeline,
    mm_abt: wgpu::ComputePipeline,
    mm_f: wgpu::ComputePipeline,
    mm_atb_f: wgpu::ComputePipeline,
    mm_abt_f: wgpu::ComputePipeline,
    bias_add: wgpu::ComputePipeline,
    relu: wgpu::ComputePipeline,
    relu_bwd: wgpu::ComputePipeline,
    xent: wgpu::ComputePipeline,
    colsum: wgpu::ComputePipeline,
    adam: wgpu::ComputePipeline,
    fast_gemm: bool,
    perfil: Option<Perfil>,
}

/// Marcas de tempo por sessão. O WebGPU limita um `QuerySet` a 4096 consultas,
/// então cabem 2048 operações — o perfilador avisa quando o plano não cabe.
const MARCAS: u32 = 4096;

impl Gpu {
    /// Abre a GPU discreta de maior desempenho disponível.
    pub fn new() -> Result<Gpu, String> {
        Gpu::with_preference(wgpu::PowerPreference::HighPerformance)
    }

    pub fn with_preference(pref: wgpu::PowerPreference) -> Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: pref,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("nenhum adaptador disponível: {e}"))?;

        let info = adapter.get_info();
        let etiqueta = format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // Pede as marcas de tempo só se o adaptador oferecer; sem elas o resto
        // funciona igual, apenas sem perfilamento.
        let tem_timestamp = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let features = if tem_timestamp {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let limites = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rtensor"),
            required_features: features,
            required_limits: limites,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            ..Default::default()
        }))
        .map_err(|e| format!("falha ao abrir o dispositivo: {e}"))?;

        let pipe = |src: &str, entry: &str, label: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let perfil = tem_timestamp.then(|| {
            let bytes = (MARCAS as u64) * 8;
            Perfil {
                conjunto: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("marcas"),
                    ty: wgpu::QueryType::Timestamp,
                    count: MARCAS,
                }),
                resolucao: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("resolucao"),
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                leitura: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("leitura"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                capacidade: MARCAS,
                periodo: queue.get_timestamp_period(),
                proximo: Cell::new(0),
                rotulos: RefCell::new(Vec::new()),
                ativo: Cell::new(false),
            }
        });

        Ok(Gpu {
            mm: pipe(kernels::MM, "mm", "mm"),
            mm_atb: pipe(kernels::MM, "mm_atb", "mm_atb"),
            mm_abt: pipe(kernels::MM, "mm_abt", "mm_abt"),
            mm_f: pipe(gemm::MM_FAST, "mm", "mm_fast"),
            mm_atb_f: pipe(gemm::MM_FAST, "mm_atb", "mm_atb_fast"),
            mm_abt_f: pipe(gemm::MM_FAST, "mm_abt", "mm_abt_fast"),
            bias_add: pipe(kernels::VEC, "bias_add", "bias_add"),
            relu: pipe(kernels::VEC, "relu", "relu"),
            relu_bwd: pipe(kernels::RELU_BWD, "relu_bwd", "relu_bwd"),
            xent: pipe(kernels::XENT, "xent", "xent"),
            colsum: pipe(kernels::COLSUM, "colsum", "colsum"),
            adam: pipe(kernels::ADAM, "adam", "adam"),
            fast_gemm: true,
            perfil,
            info: etiqueta,
            device,
            queue,
        })
    }

    /// Nome e tipo do adaptador em uso.
    pub fn info(&self) -> &str {
        &self.info
    }

    /// `true` se o adaptador oferece marcas de tempo e o perfilamento funciona.
    pub fn tem_perfilamento(&self) -> bool {
        self.perfil.is_some()
    }

    /// Quantas operações cabem numa sessão de perfilamento.
    pub fn capacidade_perfil(&self) -> usize {
        self.perfil.as_ref().map_or(0, |p| (p.capacidade / 2) as usize)
    }

    /// Escolhe entre o GEMM ladrilhado em dois níveis (padrão) e o ingênuo.
    pub fn set_fast_gemm(&mut self, ligado: bool) {
        self.fast_gemm = ligado;
    }

    pub fn fast_gemm(&self) -> bool {
        self.fast_gemm
    }

    fn pipeline(&self, k: Kernel) -> &wgpu::ComputePipeline {
        match k {
            Kernel::Mm => &self.mm,
            Kernel::MmAtb => &self.mm_atb,
            Kernel::MmAbt => &self.mm_abt,
            Kernel::MmF => &self.mm_f,
            Kernel::MmAtbF => &self.mm_atb_f,
            Kernel::MmAbtF => &self.mm_abt_f,
            Kernel::BiasAdd => &self.bias_add,
            Kernel::Relu => &self.relu,
            Kernel::ReluBwd => &self.relu_bwd,
            Kernel::Xent => &self.xent,
            Kernel::Colsum => &self.colsum,
            Kernel::Adam => &self.adam,
        }
    }

    // ---------------------------------------------------------------- buffers

    fn storage(&self, len: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Reserva um tensor zerado no dispositivo.
    pub fn zeros(&self, shape: &[usize]) -> GpuTensor {
        let len: usize = shape.iter().product();
        let t = GpuTensor { shape: shape.to_vec(), len, buf: self.storage(len, "zeros") };
        self.queue.write_buffer(&t.buf, 0, bytemuck::cast_slice(&vec![0.0f32; len.max(1)]));
        t
    }

    /// Envia um tensor da CPU para o dispositivo.
    pub fn upload(&self, t: &Tensor) -> GpuTensor {
        let g = GpuTensor {
            shape: t.shape().to_vec(),
            len: t.len(),
            buf: self.storage(t.len(), "upload"),
        };
        self.queue.write_buffer(&g.buf, 0, bytemuck::cast_slice(t.data()));
        g
    }

    fn upload_u32(&self, v: &[u32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("labels"),
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    /// Traz um tensor de volta para a CPU. Sincroniza com o dispositivo.
    pub fn download(&self, t: &GpuTensor) -> Tensor {
        let bytes = (t.len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&t.buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let dados: Vec<f32> = self.mapear(&staging, |b| bytemuck::cast_slice(b).to_vec());
        Tensor::new(&t.shape, dados)
    }

    /// Mapeia um buffer de leitura e aplica `f` aos bytes.
    fn mapear<T>(&self, buf: &wgpu::Buffer, f: impl FnOnce(&[u8]) -> T) -> T {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll falhou");
        rx.recv().expect("canal fechado").expect("map falhou");

        let vista = slice.get_mapped_range().expect("get_mapped_range falhou");
        let saida = f(&vista[..]);
        drop(vista);
        buf.unmap();
        saida
    }

    /// Espera a GPU terminar tudo que foi submetido.
    pub fn sync(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll falhou");
    }

    fn uniform<T: Pod>(&self, v: T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&v),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn bind(&self, k: Kernel, entradas: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = entradas
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline(k).get_bind_group_layout(0),
            entries: &entries,
        })
    }

    fn montar<T: Pod>(
        &self,
        kernel: Kernel,
        params: T,
        buffers: &[&wgpu::Buffer],
        gx: u32,
        gy: u32,
        rotulo: &'static str,
    ) -> Op {
        let uniforme = self.uniform(params);
        let mut todos: Vec<&wgpu::Buffer> = vec![&uniforme];
        todos.extend_from_slice(buffers);
        let bind = self.bind(kernel, &todos);
        Op { kernel, bind, uniforme, gx, gy, rotulo }
    }

    /// Grava uma operação já montada. É o caminho quente: nenhum descritor novo.
    pub fn record(&self, enc: &mut wgpu::CommandEncoder, op: &Op) {
        let marcas = self.perfil.as_ref().and_then(|p| {
            if !p.ativo.get() {
                return None;
            }
            let i = p.proximo.get();
            if i + 2 > p.capacidade {
                return None;
            }
            p.proximo.set(i + 2);
            p.rotulos.borrow_mut().push(op.rotulo);
            Some((i, i + 1))
        });

        let timestamps = marcas.map(|(ini, fim)| wgpu::ComputePassTimestampWrites {
            query_set: &self.perfil.as_ref().unwrap().conjunto,
            beginning_of_pass_write_index: Some(ini),
            end_of_pass_write_index: Some(fim),
        });

        let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op.rotulo),
            timestamp_writes: timestamps,
        });
        passe.set_pipeline(self.pipeline(op.kernel));
        passe.set_bind_group(0, &op.bind, &[]);
        passe.dispatch_workgroups(op.gx, op.gy, 1);
    }

    // ------------------------------------------------------- montagem das ops

    /// Divide `grupos` numa grade 2-D, já que cada dimensão vai até 65535.
    /// Devolve `(x, y)`; `x` é o que os kernels recebem como `gx`.
    fn grade(grupos: usize) -> (u32, u32) {
        const TETO: usize = 32768;
        if grupos <= TETO {
            (grupos as u32, 1)
        } else {
            (TETO as u32, grupos.div_ceil(TETO) as u32)
        }
    }

    fn op_mm(
        &self,
        lento: Kernel,
        rapido: Kernel,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        rotulo: &'static str,
    ) -> Op {
        let (kernel, ladrilho) = if self.fast_gemm { (rapido, 64) } else { (lento, 16) };
        self.montar(
            kernel,
            Dims { m: m as u32, n: n as u32, k: k as u32, pad: 0 },
            &[&a.buf, &b.buf, &c.buf],
            n.div_ceil(ladrilho) as u32,
            m.div_ceil(ladrilho) as u32,
            rotulo,
        )
    }

    /// `C = A · B`, com `A` em `[m, k]` e `B` em `[k, n]`.
    pub fn op_matmul(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> Op {
        let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
        self.op_mm(Kernel::Mm, Kernel::MmF, a, b, c, m, n, k, "matmul")
    }

    /// `C = Aᵀ · B`, com `A` guardada em `[k, m]` — é o `∂L/∂W = Xᵀδ`.
    pub fn op_matmul_at_b(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> Op {
        let (k, m, n) = (a.shape[0], a.shape[1], b.shape[1]);
        self.op_mm(Kernel::MmAtb, Kernel::MmAtbF, a, b, c, m, n, k, "matmul_at_b")
    }

    /// `C = A · Bᵀ`, com `B` guardada em `[n, k]` — é o `∂L/∂X = δWᵀ`.
    pub fn op_matmul_a_bt(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> Op {
        let (m, k, n) = (a.shape[0], a.shape[1], b.shape[0]);
        self.op_mm(Kernel::MmAbt, Kernel::MmAbtF, a, b, c, m, n, k, "matmul_a_bt")
    }

    /// `Z ← Z + 1ₙ b`.
    pub fn op_bias_add(&self, z: &GpuTensor, b: &GpuTensor) -> Op {
        let (gx, gy) = Self::grade(z.len.div_ceil(256));
        self.montar(
            Kernel::BiasAdd,
            P4 { a: z.len as u32, b: b.len as u32, c: gx, d: 0 },
            &[&b.buf, &z.buf],
            gx,
            gy,
            "bias_add",
        )
    }

    /// `Y = max(X, 0)`.
    pub fn op_relu(&self, x: &GpuTensor, y: &GpuTensor) -> Op {
        let (gx, gy) = Self::grade(x.len.div_ceil(256));
        self.montar(
            Kernel::Relu,
            P4 { a: x.len as u32, b: 0, c: gx, d: 0 },
            &[&x.buf, &y.buf],
            gx,
            gy,
            "relu",
        )
    }

    /// `δ ← δ ⊙ 1[Z > 0]`, in-place sobre `dz`.
    pub fn op_relu_bwd(&self, z: &GpuTensor, dz: &GpuTensor) -> Op {
        let (gx, gy) = Self::grade(z.len.div_ceil(256));
        self.montar(
            Kernel::ReluBwd,
            P4 { a: z.len as u32, b: 0, c: gx, d: 0 },
            &[&z.buf, &dz.buf],
            gx,
            gy,
            "relu_bwd",
        )
    }

    /// Softmax + entropia cruzada fundidos: perda por linha e `δ = (P − Y)/n`.
    pub fn op_softmax_xent(
        &self,
        logits: &GpuTensor,
        labels: &wgpu::Buffer,
        delta: &GpuTensor,
        loss: &GpuTensor,
    ) -> Op {
        let (rows, cols) = (logits.shape[0], logits.shape[1]);
        assert!(rows.div_ceil(64) <= 65535, "lote grande demais para um dispatch");
        self.montar(
            Kernel::Xent,
            P4 { a: rows as u32, b: cols as u32, c: 0, d: 0 },
            &[&logits.buf, labels, &delta.buf, &loss.buf],
            rows.div_ceil(64) as u32,
            1,
            "softmax_xent",
        )
    }

    /// Quantas linhas cada thread soma no primeiro estágio da redução.
    const BLOCO_REDUCAO: usize = 32;

    /// Número de parciais que o primeiro estágio produz para um lote de `rows`.
    pub fn particoes(rows: usize) -> usize {
        rows.div_ceil(Self::BLOCO_REDUCAO)
    }

    fn op_colsum_estagio(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        chunk: usize,
        rotulo: &'static str,
    ) -> Op {
        self.montar(
            Kernel::Colsum,
            P4 { a: rows as u32, b: cols as u32, c: chunk as u32, d: 0 },
            &[src, dst],
            cols.div_ceil(64) as u32,
            rows.div_ceil(chunk) as u32,
            rotulo,
        )
    }

    /// `dst = 1ₙᵀ src`, em um ou dois estágios conforme o número de linhas.
    pub fn ops_colsum(&self, src: &GpuTensor, dst: &GpuTensor, scratch: &GpuTensor) -> Vec<Op> {
        let (rows, cols) = (src.shape[0], src.shape[1]);
        if rows <= Self::BLOCO_REDUCAO {
            return vec![self.op_colsum_estagio(&src.buf, &dst.buf, rows, cols, rows, "colsum")];
        }
        let partes = Self::particoes(rows);
        vec![
            self.op_colsum_estagio(
                &src.buf,
                &scratch.buf,
                rows,
                cols,
                Self::BLOCO_REDUCAO,
                "colsum_1",
            ),
            self.op_colsum_estagio(&scratch.buf, &dst.buf, partes, cols, partes, "colsum_2"),
        ]
    }

    /// Passo do Adam fundido. O uniforme é reescrito a cada passo por
    /// [`Gpu::atualizar_adam`], sem refazer o bind group.
    pub fn op_adam(
        &self,
        theta: &GpuTensor,
        m: &GpuTensor,
        v: &GpuTensor,
        g: &GpuTensor,
        lr: f32,
        betas: (f32, f32),
    ) -> Op {
        let (gx, gy) = Self::grade(theta.len.div_ceil(256));
        let (b1, b2) = betas;
        self.montar(
            Kernel::Adam,
            AdamP {
                n: theta.len as u32,
                lr,
                b1,
                b2,
                eps: 1e-8,
                c1: 1.0 - b1,
                c2: 1.0 - b2,
                gx,
            },
            &[&g.buf, &theta.buf, &m.buf, &v.buf],
            gx,
            gy,
            "adam",
        )
    }

    /// Reescreve a correção de viés do Adam para o passo `t`.
    pub fn atualizar_adam(&self, op: &Op, n: usize, lr: f32, betas: (f32, f32), t: i32) {
        let (b1, b2) = betas;
        let (gx, _) = Self::grade(n.div_ceil(256));
        self.queue.write_buffer(
            &op.uniforme,
            0,
            bytemuck::bytes_of(&AdamP {
                n: n as u32,
                lr,
                b1,
                b2,
                eps: 1e-8,
                c1: 1.0 - b1.powi(t),
                c2: 1.0 - b2.powi(t),
                gx,
            }),
        );
    }

    // ------------------------------------------------------- atalhos imediatos

    /// `C = A · B`, montando e gravando de uma vez. Conveniente para uso
    /// avulso; num laço quente prefira [`Gpu::op_matmul`] + [`Gpu::record`].
    pub fn matmul(&self, enc: &mut wgpu::CommandEncoder, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        let op = self.op_matmul(a, b, c);
        self.record(enc, &op);
    }

    pub fn matmul_at_b(&self, enc: &mut wgpu::CommandEncoder, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        let op = self.op_matmul_at_b(a, b, c);
        self.record(enc, &op);
    }

    pub fn matmul_a_bt(&self, enc: &mut wgpu::CommandEncoder, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        let op = self.op_matmul_a_bt(a, b, c);
        self.record(enc, &op);
    }

    pub fn bias_add(&self, enc: &mut wgpu::CommandEncoder, z: &GpuTensor, b: &GpuTensor) {
        let op = self.op_bias_add(z, b);
        self.record(enc, &op);
    }

    pub fn relu(&self, enc: &mut wgpu::CommandEncoder, x: &GpuTensor, y: &GpuTensor) {
        let op = self.op_relu(x, y);
        self.record(enc, &op);
    }

    pub fn relu_bwd(&self, enc: &mut wgpu::CommandEncoder, z: &GpuTensor, dz: &GpuTensor) {
        let op = self.op_relu_bwd(z, dz);
        self.record(enc, &op);
    }

    pub fn colsum(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &GpuTensor,
        dst: &GpuTensor,
        scratch: &GpuTensor,
    ) {
        for op in self.ops_colsum(src, dst, scratch) {
            self.record(enc, &op);
        }
    }

    /// Grava várias operações **num único compute pass**.
    ///
    /// Abrir um passe por dispatch custa caro: o perfilador mostrou 40% do
    /// passo de treino fora dos kernels numa rede de 13 camadas (128
    /// dispatches). Dentro de um mesmo passe o WebGPU garante ordem de execução
    /// e visibilidade das escritas entre dispatches consecutivos, então a
    /// semântica não muda — só o número de passes.
    ///
    /// As marcas de tempo são por passe, então com o perfilamento ligado cada
    /// operação continua no seu próprio passe, para não perder a granularidade.
    pub fn record_many(&self, enc: &mut wgpu::CommandEncoder, ops: &[&Op]) {
        let perfilando = self.perfil.as_ref().is_some_and(|p| p.ativo.get());
        if perfilando {
            for op in ops {
                self.record(enc, op);
            }
            return;
        }

        let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("plano"),
            timestamp_writes: None,
        });
        for op in ops {
            passe.set_pipeline(self.pipeline(op.kernel));
            passe.set_bind_group(0, &op.bind, &[]);
            passe.dispatch_workgroups(op.gx, op.gy, 1);
        }
    }

    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    pub fn submit(&self, enc: wgpu::CommandEncoder) {
        self.queue.submit(Some(enc.finish()));
    }

    // ------------------------------------------------------------ perfilamento

    /// Roda `f` com as marcas de tempo ligadas e devolve o tempo de cada kernel
    /// **medido no dispositivo**, agregado por rótulo: `(rótulo, chamadas, ms)`.
    ///
    /// Devolve vazio se o adaptador não oferece `TIMESTAMP_QUERY`.
    pub fn perfilar<F: FnOnce()>(&self, f: F) -> Vec<(&'static str, u32, f64)> {
        let Some(p) = self.perfil.as_ref() else { return Vec::new() };

        p.proximo.set(0);
        p.rotulos.borrow_mut().clear();
        p.ativo.set(true);
        f();
        p.ativo.set(false);

        let n = p.proximo.get();
        if n == 0 {
            return Vec::new();
        }

        let mut enc = self.encoder();
        enc.resolve_query_set(&p.conjunto, 0..n, &p.resolucao, 0);
        enc.copy_buffer_to_buffer(&p.resolucao, 0, &p.leitura, 0, (n as u64) * 8);
        self.queue.submit(Some(enc.finish()));

        let tiques: Vec<u64> = self.mapear(&p.leitura, |b| bytemuck::cast_slice(b).to_vec());
        let rotulos = p.rotulos.borrow();

        let mut agregado: Vec<(&'static str, u32, f64)> = Vec::new();
        for (i, rotulo) in rotulos.iter().enumerate() {
            let (ini, fim) = (tiques[i * 2], tiques[i * 2 + 1]);
            // O contador pode dar a volta entre passes; nesse caso descartamos.
            let ms = if fim >= ini {
                (fim - ini) as f64 * p.periodo as f64 / 1e6
            } else {
                0.0
            };
            match agregado.iter_mut().find(|(r, _, _)| r == rotulo) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += ms;
                }
                None => agregado.push((rotulo, 1, ms)),
            }
        }
        agregado.sort_by(|a, b| b.2.total_cmp(&a.2));
        agregado
    }
}

// ---------------------------------------------------------------------- modelo

struct CamadaGpu {
    w: GpuTensor,
    b: GpuTensor,
    mw: GpuTensor,
    vw: GpuTensor,
    mb: GpuTensor,
    vb: GpuTensor,
    dw: GpuTensor,
    db: GpuTensor,
    dbp: GpuTensor,
    z: GpuTensor,
    a: GpuTensor,
    delta: GpuTensor,
}

/// MLP treinada inteiramente na GPU, com `relu` nas camadas ocultas e
/// softmax + entropia cruzada na saída.
///
/// O plano de execução — todas as [`Op`] do passo — é montado uma única vez na
/// construção. Cada `step` copia o lote para os buffers de entrada, atualiza os
/// uniformes do Adam e regrava o plano.
pub struct GpuMlp {
    camadas: Vec<CamadaGpu>,
    loss: GpuTensor,
    /// Entrada fixa que o plano referencia; cada passo copia o lote para cá.
    xin: GpuTensor,
    rotulos: wgpu::Buffer,
    ops_fwd: Vec<Op>,
    ops_bwd: Vec<Op>,
    ops_adam: Vec<Op>,
    /// `(n, lr)` de cada op do Adam, para reescrever o uniforme por passo.
    adam_dims: Vec<usize>,
    lote: usize,
    t: i32,
    lr: f32,
}

impl GpuMlp {
    /// Constrói a partir dos pesos de um modelo de CPU — `[W¹, b¹, W², b², …]`,
    /// a ordem devolvida por [`crate::nn::Sequential::params`].
    pub fn from_params(gpu: &Gpu, params: &[Param], lote: usize, lr: f32) -> GpuMlp {
        assert!(params.len() % 2 == 0, "esperado pares (W, b)");
        assert!(!params.is_empty(), "modelo sem camadas");

        let camadas: Vec<CamadaGpu> = params
            .chunks(2)
            .map(|par| {
                let (w, b) = (par[0].value().clone(), par[1].value().clone());
                let (entradas, unidades) = (w.shape()[0], w.shape()[1]);
                CamadaGpu {
                    mw: gpu.zeros(&[entradas, unidades]),
                    vw: gpu.zeros(&[entradas, unidades]),
                    mb: gpu.zeros(&[1, unidades]),
                    vb: gpu.zeros(&[1, unidades]),
                    dw: gpu.zeros(&[entradas, unidades]),
                    db: gpu.zeros(&[1, unidades]),
                    dbp: gpu.zeros(&[Gpu::particoes(lote), unidades]),
                    z: gpu.zeros(&[lote, unidades]),
                    a: gpu.zeros(&[lote, unidades]),
                    delta: gpu.zeros(&[lote, unidades]),
                    w: gpu.upload(&w),
                    b: gpu.upload(&b),
                }
            })
            .collect();

        let entradas = params[0].shape()[0];
        let xin = gpu.zeros(&[lote, entradas]);
        let rotulos = gpu.upload_u32(&vec![0u32; lote]);
        let loss = gpu.zeros(&[lote]);

        let ultima = camadas.len() - 1;

        // ------------------------------------------------------------ avanço
        let mut ops_fwd = Vec::new();
        for i in 0..=ultima {
            let entrada: &GpuTensor = if i == 0 { &xin } else { &camadas[i - 1].a };
            let c = &camadas[i];
            ops_fwd.push(gpu.op_matmul(entrada, &c.w, &c.z));
            ops_fwd.push(gpu.op_bias_add(&c.z, &c.b));
            if i < ultima {
                ops_fwd.push(gpu.op_relu(&c.z, &c.a));
            }
        }

        // --------------------------------------------------- perda e backward
        let mut ops_bwd = vec![gpu.op_softmax_xent(
            &camadas[ultima].z,
            &rotulos,
            &camadas[ultima].delta,
            &loss,
        )];
        for i in (0..=ultima).rev() {
            let entrada: &GpuTensor = if i == 0 { &xin } else { &camadas[i - 1].a };
            let c = &camadas[i];
            ops_bwd.push(gpu.op_matmul_at_b(entrada, &c.delta, &c.dw));
            ops_bwd.extend(gpu.ops_colsum(&c.delta, &c.db, &c.dbp));
            if i > 0 {
                let anterior = &camadas[i - 1];
                ops_bwd.push(gpu.op_matmul_a_bt(&c.delta, &c.w, &anterior.delta));
                ops_bwd.push(gpu.op_relu_bwd(&anterior.z, &anterior.delta));
            }
        }

        // --------------------------------------------------------------- Adam
        let mut ops_adam = Vec::new();
        let mut adam_dims = Vec::new();
        for c in &camadas {
            ops_adam.push(gpu.op_adam(&c.w, &c.mw, &c.vw, &c.dw, lr, (0.9, 0.999)));
            adam_dims.push(c.w.len);
            ops_adam.push(gpu.op_adam(&c.b, &c.mb, &c.vb, &c.db, lr, (0.9, 0.999)));
            adam_dims.push(c.b.len);
        }

        GpuMlp {
            camadas,
            loss,
            xin,
            rotulos,
            ops_fwd,
            ops_bwd,
            ops_adam,
            adam_dims,
            lote,
            t: 0,
            lr,
        }
    }

    /// Copia o lote para os buffers que o plano referencia. É uma cópia
    /// dispositivo→dispositivo, sem passar pelo host.
    fn carregar(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: &GpuTensor, labels: &wgpu::Buffer) {
        assert_eq!(
            x.len, self.xin.len,
            "lote com {} elementos, plano montado para {}",
            x.len, self.xin.len
        );
        let _ = gpu;
        enc.copy_buffer_to_buffer(&x.buf, 0, &self.xin.buf, 0, (self.xin.len * 4) as u64);
        enc.copy_buffer_to_buffer(labels, 0, &self.rotulos, 0, (self.lote * 4) as u64);
    }

    /// Um passo completo: avanço, retropropagação e Adam, gravados num único
    /// `CommandEncoder` e submetidos sem sincronizar.
    pub fn step(&mut self, gpu: &Gpu, x: &GpuTensor, labels: &wgpu::Buffer) {
        self.t += 1;
        for (op, &n) in self.ops_adam.iter().zip(&self.adam_dims) {
            gpu.atualizar_adam(op, n, self.lr, (0.9, 0.999), self.t);
        }

        let mut enc = gpu.encoder();
        self.carregar(gpu, &mut enc, x, labels);
        let plano: Vec<&Op> =
            self.ops_fwd.iter().chain(&self.ops_bwd).chain(&self.ops_adam).collect();
        gpu.record_many(&mut enc, &plano);
        gpu.submit(enc);
    }

    /// Só o avanço — o caminho de inferência.
    pub fn forward_only(&self, gpu: &Gpu, x: &GpuTensor) {
        let mut enc = gpu.encoder();
        assert_eq!(x.len, self.xin.len, "lote diferente do plano");
        enc.copy_buffer_to_buffer(&x.buf, 0, &self.xin.buf, 0, (self.xin.len * 4) as u64);
        let plano: Vec<&Op> = self.ops_fwd.iter().collect();
        gpu.record_many(&mut enc, &plano);
        gpu.submit(enc);
    }

    /// Logits do último avanço, baixados para a CPU.
    pub fn logits(&self, gpu: &Gpu) -> Tensor {
        gpu.download(&self.camadas[self.camadas.len() - 1].z)
    }

    /// Perda média do último passo. Sincroniza com o dispositivo.
    pub fn last_loss(&self, gpu: &Gpu) -> f32 {
        gpu.download(&self.loss).mean_all()
    }

    /// Baixa os pesos para a CPU, na ordem `[W¹, b¹, W², b², …]`.
    pub fn weights(&self, gpu: &Gpu) -> Vec<Tensor> {
        let mut out = Vec::new();
        for c in &self.camadas {
            out.push(gpu.download(&c.w));
            out.push(gpu.download(&c.b));
        }
        out
    }

    /// Gradientes do último passo, na mesma ordem dos pesos.
    pub fn grads(&self, gpu: &Gpu) -> Vec<Tensor> {
        let mut out = Vec::new();
        for c in &self.camadas {
            out.push(gpu.download(&c.dw));
            out.push(gpu.download(&c.db));
        }
        out
    }

    pub fn batch_size(&self) -> usize {
        self.lote
    }

    /// Número de dispatches por passo de treino — a métrica que o custo de
    /// montagem de descritores costumava seguir.
    pub fn dispatches_por_passo(&self) -> usize {
        self.ops_fwd.len() + self.ops_bwd.len() + self.ops_adam.len()
    }
}

/// Envia rótulos inteiros para o dispositivo.
pub fn upload_labels(gpu: &Gpu, labels: &[usize]) -> wgpu::Buffer {
    let v: Vec<u32> = labels.iter().map(|&l| l as u32).collect();
    gpu.upload_u32(&v)
}
