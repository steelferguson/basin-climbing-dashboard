Creating a dashboard for Basin Climbing and Fitness

* We are using Capitan which has reports but no dashboard
* Pull transaction level data from both Square and Stripe (Capitan uses both currently)
* Also pulls data from the Capitan APIs since there is additional data available there
* Creates a cleaned dataframe with additional columns for categorization and saves that
* Then uses the dataframe to create dynamic visualization
* This is then hosted on Heroku

Note: This was created as a minimal viable product with the plan to come back soon to refine
