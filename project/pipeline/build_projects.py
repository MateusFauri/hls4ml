import os
import yaml
import pandas as pd
import hls4ml

CONFIG_DIR = "configs"

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

results = []

def run(bits, experiment):

    for bit in bits:

        print("=" * 80)
        print(f"Building {bit} bits")
        print("=" * 80)

        yaml_path = os.path.join(
            CONFIG_DIR,
            f"{experiment}_{bit}b.yaml"
        )

        if not os.path.exists(yaml_path):

            print("Configuration not found.\n")
            continue

        with open(yaml_path) as f:

            config = yaml.safe_load(f)

        print("Loading HLS project...")

        hls_model = hls4ml.converters.keras_to_hls(config)

        print("Running synthesis...")

        hls_model.build(
            reset=False,
            csim=True,
            synth=True,
            cosim=False,
            validation=False,
            export=True
        )

        report = hls4ml.report.read_vivado_report(
            config["OutputDir"]
        )

        latency = None
        ii = None
        bram = None
        dsp = None
        ff = None
        lut = None

        try:
            latency = report["CSynthesisReport"]["EstimatedLatency"]
            ii = report["CSynthesisReport"]["IntervalMin"]
            bram = report["CSynthesisReport"]["BRAM_18K"]
            dsp = report["CSynthesisReport"]["DSP48E"]
            ff = report["CSynthesisReport"]["FF"]
            lut = report["CSynthesisReport"]["LUT"]
        except Exception:
            print("Unable to read synthesis report.")

        results.append({
            "bits": bit,
            "latency": latency,
            "ii": ii,
            "bram": bram,
            "dsp": dsp,
            "ff": ff,
            "lut": lut
        })

        print()

    report_df = pd.DataFrame(results)

    report_df.to_csv(

        os.path.join(

            REPORT_DIR,

            f"{experiment}_build_report.csv"

        ),

        index=False

    )

    print("=" * 80)
    print("Finished.")
    print("=" * 80)

    print(report_df)


def main():

    run(
        bits=[8,4,2,1],
        experiment="t1"
    )


if __name__ == "__main__":
    main()