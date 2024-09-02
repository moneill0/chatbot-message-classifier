## Project Guidelines

This repo contains 2 data files: `labeled_messages.csv` and `empathies.csv`. The `labeled_messages.csv` file has 3500 user messages, and empathy labels for each one (which correspond to the empathy responses that Woebot gives to that message). There are 62 empathy labels. The other CSV is of the empathies and their polarities, from -1 to 1. 

Program a classifier that can output the empathy for each message. Note that some rows are labeled with multiple classes, so your classifier needs to output multiple empathy classes for those cases. It is most important that your classifier outputs an empathy of the same polarity as the target empathy (positive/negative/neutral), but it is optimal if it outputs exactly the same empathy/empathies. You may want to create separate classifiers for empathy and polarity and compare them to each other.

Come up with one or more metrics to evaluate your algorithm, and be prepared to discuss ideas for improving it.

Discuss things you would do if you had more time.

Your submission will be evaluated in the following areas:

1. Data exploration, visualization, pre-processing, and cleanup
1. Model selection and/or use of multiple models
1. Choice and discussion of evaluation metrics 
1. Overall code design, cleanliness, clarity of comments
1. Discussion of approach, and list of things you'd do if you had more time

Guidelines:

- Please complete this exercise in Python, and include instructions on usage in your submission. 
  - A big purpose of this project is to see how you write and organize your code. 
- We have no specific time bounds defined for this problem, but we understand that this is an extra time commitment for you, so please keep in mind that we are not looking for this to take a significant chunk of time out of your day! 😊
