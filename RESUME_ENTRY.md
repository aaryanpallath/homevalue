# homevalue — Home Value Estimates for King County, WA
Tech: Python, scikit-learn, pandas, FastAPI, JavaScript, pytest, GitHub Actions
- Built a home valuation model and web app on 114,839 King County Assessor sales, cleaning out non-market transfers and serving estimates, calibrated price ranges and comparable sales for 451,612 homes via FastAPI.
- Reached 7.5% median error on 6,829 later sales (61% within 10%), beating a comparable-sales baseline at 14.4% and a hedonic regression at 9.4%, with leakage-free time-based features.
- Found ranges under-covered expensive homes (69% vs 80%) and fixed it with price-band conformal calibration; caught a 2020 deed-code change that silently dropped 90% of post-2019 sales.
