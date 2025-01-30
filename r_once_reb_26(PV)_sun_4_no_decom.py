import os
import datetime
import joblib
import DER
from common import create_simulation, START_DATE, FINISH_DATE

if __name__ == "__main__":
    filename = os.path.splitext(os.path.basename(__file__))[0]
    print(filename)
    # Setup the folder
    dirname = os.path.join(
        os.path.dirname(__file__), "tmp"
    )
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # Run Ramsey Once
    s = create_simulation(
        start_datetime=START_DATE,
        finish_datetime=FINISH_DATE,
    )
    s.calculate_ramsey_prices = True
    s.calculate_ramsey_prices_once = True
    s.use_tva_adoption_equations = True
    s.calculate_ramsey_prices_24_hours = False

    for l in s.lses:
        l.sun_hours_per_day = 4
        l.dg_investment_discount = 0
        l.dg_pv_only_investment_discount = 0.26

    # No Decom. Generators
    for g in s.generators:
        g.retirement_date = None

    s.setup()

    try:
        s.run()
    except Exception as e:
        print(e)
    joblib.dump(s, filename=os.path.join(dirname, f"{filename}.gz"), compress=5)
