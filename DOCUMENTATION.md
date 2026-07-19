# Simply-Fit — Project Documentation

## What this is

Simply-Fit is a weight management system built around one observation: 
people stop logging food within a week. The app removes that requirement 
entirely. The user logs their weight once a day. Everything else is inferred.

The daily weight reading is the net result of everything the body consumed 
and burned. If you can extract the signal from the noise in that number, 
you do not need a food log. That is the premise this project is built on.

---

## How it evolved from HealthLens

HealthLens was my first version of this idea. It worked, but it was 
essentially a calculator — you entered your goal, it returned a fixed 
calorie target, and that number never changed regardless of what was 
actually happening to your weight.

The core problem with that approach is the direction of the calculation. 
HealthLens said: here is your deficit, this is what should happen. 
Simply-Fit flips this. It observes what is actually happening and works 
backward to infer the calorie balance. Same 7700 kcal/kg constant, 
opposite direction.

The practical difference is significant. HealthLens gave the same 
recommendation on day 1 and day 30. Simply-Fit's recommendation updates 
every day based on the real trend in the user's weight data.

A second problem with HealthLens was that it treated every weight reading 
as equally valid. A 1.5 kg spike from a salty dinner would shift the 
calculations the same way a real fat gain would. Simply-Fit filters these 
out before doing any calculation.

---

## What is actually different under the hood

**Noise filtering.** Raw daily weight is too noisy to act on directly. 
Before any calculation runs, the weight log passes through an Exponential 
Weighted Moving Average filter that separates the true fat-change trend 
from daily water and glycogen fluctuation.

**Backward calorie inference.** A linear regression is fitted on the 
smoothed trend. The slope gives kg change per day. Multiplied by 7700, 
this is the user's actual daily calorie balance — inferred from their 
body, not entered manually.

**Anomaly detection.** Isolation Forest runs on the residuals between raw 
readings and the smoothed trend. Flagged readings are excluded from the 
calorie calculation. HealthLens had none of this.

**Personalised coefficient.** After 14 days of data, the regression gives 
a personal calorie-per-kg value for that specific user. HealthLens used 
7700 for everyone. Simply-Fit learns the individual.

**LSTM forecasting.** A neural network trained on 500 synthetic users 
forecasts the next 7 days of weight trajectory. It captures patterns that 
linear projection cannot — plateaus, metabolic slowdown, weekly 
fluctuation cycles.

**AI coach.** The user can ask questions about their progress. Before 
each response, the system injects the user's actual metrics — weight 
trend, inferred balance, goal, disease conditions — into the prompt. The 
coach answers from real data, not generic advice.

---

## Data and validation

There is no large public dataset of daily weight logs with disease 
conditions, so I generated synthetic training data — 500 users across 
90 days each, using real physiological equations for BMR, TDEE, and 
noise parameters from clinical weight variability literature.

After generating the data I validated it against the NHANES public health 
survey using the Kolmogorov-Smirnov test. This revealed two biases — 
synthetic users were 7 kg heavier and 10 years younger on average than 
the real population. I recalibrated the generator to sample from 
NHANES-derived distributions. Weight KS statistic improved by 62% and 
age KS statistic improved by 75%.

This validation step matters because it shows the training data reflects 
real human physiology rather than arbitrary assumptions.

---

## Known limitations

The LSTM is trained on synthetic data. It performs well within the 
distribution it was trained on but may not fully generalise to real users 
with illness, medication effects, or hormonal variability that synthetic 
data cannot capture. The personalised regression layer partially 
compensates for this by adapting to each user's individual data over time.

The AI coach is currently using a rule-based mock response while API 
credits are obtained. The integration is architecturally complete — 
activating the real Gemini API requires uncommenting two lines of code.

---

## Tech stack

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Streamlit, 
Google Gemini API, NHANES dataset.

---

## How to run

git clone https://github.com/tyagiji369/Simply-Fit.git
cd Simply-Fit
python -m venv venv
source venv/bin/activate
pip install streamlit pandas numpy scikit-learn matplotlib tensorflow
pip install google-genai python-dotenv
streamlit run app/streamlit_app.py

Not a substitute for professional medical advice.