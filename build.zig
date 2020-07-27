const Builder = @import("std").build.Builder;

pub fn build(b: *Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("resources", "src/main.zig");
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());

    const test_cmd = b.addTest("src/main.zig");
    const test_step = b.step("test", "run project tests");
    test_step.dependOn(&test_cmd.step);

    const git_commit_cmd = b.addSystemCommand(&[_][]const u8{ "git", "commit" });
    const git_commit_step = b.step("commit", "perform a git commit");
    git_commit_step.dependOn(&test_cmd.step);
    git_commit_step.dependOn(&git_commit_cmd.step);

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
