//! Lista os adaptadores e o que cada um oferece — features e limites que
//! decidem quais otimizações são possíveis.

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

    for a in &adapters {
        let info = a.get_info();
        println!("\n=== {} | {:?} | {:?} ===", info.name, info.device_type, info.backend);

        let f = a.features();
        let interessantes = [
            ("SHADER_F16", wgpu::Features::SHADER_F16),
            ("SUBGROUP", wgpu::Features::SUBGROUP),
            ("SUBGROUP_BARRIER", wgpu::Features::SUBGROUP_BARRIER),
            ("PIPELINE_STATISTICS_QUERY", wgpu::Features::PIPELINE_STATISTICS_QUERY),
            ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
            ("MAPPABLE_PRIMARY_BUFFERS", wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            ("BUFFER_BINDING_ARRAY", wgpu::Features::BUFFER_BINDING_ARRAY),
            ("EXPERIMENTAL_COOPERATIVE_MATRIX", wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
        ];
        for (nome, bit) in interessantes {
            println!("  {:<28} {}", nome, if f.contains(bit) { "sim" } else { "não" });
        }

        let l = a.limits();
        println!("  ---");
        println!("  max_buffer_size                {} MB", l.max_buffer_size / 1024 / 1024);
        println!("  max_storage_buffer_binding     {} MB", l.max_storage_buffer_binding_size as u64 / 1024 / 1024);
        println!("  max_storage_buffers_per_stage  {}", l.max_storage_buffers_per_shader_stage);
        println!("  max_compute_workgroup_storage  {} KB", l.max_compute_workgroup_storage_size / 1024);
        println!("  max_invocations_per_workgroup  {}", l.max_compute_invocations_per_workgroup);
        println!("  max_workgroups_per_dimension   {}", l.max_compute_workgroups_per_dimension);
        println!("  min_uniform_offset_alignment   {}", l.min_uniform_buffer_offset_alignment);
    }
}
