# Simply-Fit

A personal weight management system built around one core idea — the user should not have to log their food. Daily weight is already the net result of everything the body consumed and burned. Simply-Fit extracts that signal and builds everything else from it.

## Background

Food logging is the standard approach in most diet apps. The problem is that people stop doing it within days. Studies on self-reported calorie intake consistently show large systematic errors even when people try their best. Simply-Fit sidesteps this entirely by treating the scale as a passive sensor and inferring calorie balance from the weight trend directly.

The mathematical basis for this is the energy balance equation — a kilogram of fat tissue represents roughly 7700 kcal. If your smoothed weight trend drops 0.35 kg over seven days, the app infers a deficit of approximately 385 kcal per day without you ever opening a food log.

## What it does

The user logs one number each morning — their weight. From that the system runs a noise filtering step to separate real fat change from daily water and glycogen fluctuations, fits a personal regression to estimate the user's actual calorie balance, flags anomalous readings that would corrupt the trend, forecasts the next two weeks using an LSTM model, and answers questions through a conversational AI coach that has access to the user's real data.

## Status

Active development. Built as part of a data science internship.
