## Welcome to My Homepage
<br> <img src="https://www.mdanderson.org/content/mda/en/research/departments-labs-institutes/labs/court-laboratory/lab-members/jcr:content/mainparsys/columns/column1/textimage_924724179.resize.jpg/1522770082405.jpg" alt="alt text" title="Title"  height="160" width="160" />
### Joonsang Lee, Ph.D. 
<button name="button"><a href="https://aznets.github.io/jslee/"> CV </a> <br>

I am a research engineer in the Department of Radiation Physics at the M.D. Anderson Cancer Center.
I received his Ph.D. in the Physics and Astronomy department at the University of Georgia. My Ph.D. research focused primarily on development of a pharmacokinetic modeling in dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) and the development of image-processing algorithms for tumor segmentation. My research as a postdoctoral follow in Bioinformatics & Computational Biology at the MD Anderson Cancer Center has focused primarily on image processing on brain tumor images with various statistical techniques and machine learning. During this postdoctoral training, my projects ranged from texture analysis for the parametric map of DSC-MRI to the analysis of tumor heterogeneity in multiparametric MRI data.

My current research is on decoding tumor phenotype with Radiomics approach such as the extraction and analysis of advanced quantitative imaging features obtained from CT, PET or MRI. I am currently funded by the Center for Radiation Oncology (CROR) to provide radiomics research support to researchers throughout the institution.

**M.D. Anderson Cancer Center** <br>
_Department of Radiation Physics_ <br>
<button name="button"><a href="https://www.mdanderson.org/research/departments-labs-institutes/labs/court-laboratory/lab-members.html"> Court Lab </a>

---
<br>
## Research Interest - ML & DL
### Machine Learning
<br><center> <img src="https://www.uruit.com/blog/wp-content/uploads/2018/02/Diagram-1-1024x435.png" alt="alt text" title="Title" /> </center>

<br>
### How Machine Learning Works
Machine learning uses two types of techniques: supervised learning, which trains a model on known input and output data so that it can predict future outputs, and unsupervised learning, which finds hidden patterns or intrinsic structures in input data.

<br><center><img src="https://www.mathworks.com/content/mathworks/www/en/discovery/machine-learning/jcr:content/mainParsys3/discoverysubsection_1965078453/mainParsys3/image_2128876021_cop.adapt.full.high.svg/1531721829647.svg" alt="alt text" title="Title" /> </center> <br>

<center> Figure 1. Machine learning techniques include both unsupervised and supervised learning.</center><br>
<br>
### <span style="color:orange"> Supervised Learning  </span>
Supervised machine learning builds a model that makes predictions based on evidence in the presence of uncertainty. A supervised learning algorithm takes a known set of input data and known responses to the data (output) and trains a model to generate reasonable predictions for the response to new data. Use supervised learning if you have known data for the output you are trying to predict.

Supervised learning uses classification and regression techniques to develop predictive models.

**Classification techniques** predict discrete responses—for example, whether an email is genuine or spam, or whether a tumor is cancerous or benign. Classification models classify input data into categories. Typical applications include medical imaging, speech recognition, and credit scoring.

Use classification if your data can be tagged, categorized, or separated into specific groups or classes. For example, applications for hand-writing recognition use classification to recognize letters and numbers. In image processing and computer vision, unsupervised pattern recognition techniques are used for object detection and image segmentation.

Common algorithms for performing classification include support vector machine (SVM), boosted and bagged decision trees, k-nearest neighbor, Naïve Bayes, discriminant analysis, logistic regression, and neural networks.

**Regression techniques** predict continuous responses—for example, changes in temperature or fluctuations in power demand. Typical applications include electricity load forecasting and algorithmic trading.

Use regression techniques if you are working with a data range or if the nature of your response is a real number, such as temperature or the time until failure for a piece of equipment.

Common regression algorithms include linear model, nonlinear model, regularization, stepwise regression, boosted and bagged decision trees, neural networks, and adaptive neuro-fuzzy learning.


### <span style="color:orange"> Unsupervised Learning  </span>
Unsupervised learning finds hidden patterns or intrinsic structures in data. It is used to draw inferences from datasets consisting of input data without labeled responses.

**Clustering** is the most common unsupervised learning technique. It is used for exploratory data analysis to find hidden patterns or groupings in data. Applications for cluster analysis include gene sequence analysis, market research, and object recognition.

For example, if a cell phone company wants optimize the locations where they build cell phone towers, they can use machine learning to estimate the number of clusters of people relying on their towers. A phone can only talk to one tower at a time, so the team uses clustering algorithms to design the best placement of cell towers to optimize signal reception for groups, or clusters, of their customers.

Common algorithms for performing clustering include k-means and k-medoids, hierarchical clustering, Gaussian mixture models, hidden Markov models, self-organizing maps, fuzzy c-means clustering, and subtractive clustering.

<br><center><img src="https://www.mathworks.com/content/mathworks/www/en/discovery/machine-learning/jcr:content/mainParsys3/discoverysubsection_1965078453/mainParsys3/image_792810770_copy.adapt.full.high.svg/1531721829837.svg" alt="alt text" title="Title" /> </center><br>
<center> Figure 2. Clustering finds hidden patterns in your data.</center><br>

How Do You Decide Which Machine Learning Algorithm to Use?
Choosing the right algorithm can seem overwhelming—there are dozens of supervised and unsupervised machine learning algorithms, and each takes a different approach to learning.

There is no best method or one size fits all. Finding the right algorithm is partly just trial and error—even highly experienced data scientists can’t tell whether an algorithm will work without trying it out. But algorithm selection also depends on the size and type of data you’re working with, the insights you want to get from the data, and how those insights will be used.

<br><center><img src="https://www.mathworks.com/content/mathworks/www/en/discovery/machine-learning/jcr:content/mainParsys3/discoverysubsection_1965078453/mainParsys3/image_2109075398_cop.adapt.full.high.svg/1531721829860.svg" alt="alt text" title="Title" /> </center><br>
<center> Figure 3. Machine learning techniques. </center><br>
Here are some guidelines on choosing between supervised and unsupervised machine learning:

Choose supervised learning if you need to train a model to make a prediction--for example, the future value of a continuous variable, such as temperature or a stock price, or a classification—for example, identify makes of cars from webcam video footage.
Choose unsupervised learning if you need to explore your data and want to train a model to find a good internal representation, such as splitting data up into clusters.
