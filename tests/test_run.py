from click.testing import CliRunner

from stable_diffusion_with_upscaler.run import main


def test_main(monkeypatch):
    monkeypatch.setattr("stable_diffusion_with_upscaler.run.run_model", lambda *_: 0)

    runner = CliRunner()
    result = runner.invoke(main, [""])
    assert result.exit_code == 2  # usage error

    result = runner.invoke(main, ["--prompt", "my cool prompt"])
    assert result.exit_code == 0
