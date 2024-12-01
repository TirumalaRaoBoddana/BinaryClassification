import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
st.sidebar.image("binary.png")
st.sidebar.header("Binary Classification Problem")
menu_option=st.sidebar.radio(label="Select one",options=["Perceptron Leanrning ALgorithm"])
if(menu_option=="Perceptron Leanrning ALgorithm"):
    st.subheader("What is Binary Classification?")
    st.write("Binary classification is a type of supervised learning problem where the task is to categorize data into two distinct classes. Each data point is labeled as belonging to one of two categories, often represented as:")
    st.markdown("""<ul style='font-weight:bold'>
                <li>0 or 1</li>
                <li>True or False</li>
                <li>Positive or Negative</li>
                </ul>""",unsafe_allow_html=True)
    st.markdown("<h6 style='font-weight:bold'>Examples of Binary Classification Problems:</h6>",unsafe_allow_html=True)
    st.markdown("""<ul>
                    <li>Spam detection: Email is classified as "spam" or "not spam."</li>
                    <li>Sentiment analysis: Text is classified as having a "positive" or "negative" sentiment.</li>
                    <li>Medical diagnosis: A test result is classified as "positive" or "negative" for a disease.</li>
                </ul>
                <h4>Dataset:</h4>
                """,unsafe_allow_html=True)
    st.code("""
            import pandas as pd
            import numpy as np
            from sklearn.datasets import make_classification
            import matplotlib.pyplot as plt

            # Generate the dataset
            X, y = make_classification(
                n_samples=100,
                n_features=2,
                n_informative=1,
                n_redundant=0,
                n_clusters_per_class=1,
                n_classes=2,
                random_state=41,
                class_sep=0.8
            )

            def get_dataset():
                return X, y

            # Plotting the data
            X, y = get_dataset()
            plt.figure(figsize=(8, 6))
            plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
            plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Binary Classification Dataset')
            plt.legend()
            plt.grid(True)
            plt.show()           
            """)
    st.markdown("""<h4>
                    Output:
                </h4>""",unsafe_allow_html=True)
    st.image("classification.jpg", caption="Sample Dataset Visualization", use_column_width=True)
    st.subheader("Perceptron Algorithm for Binary Classification")
    st.write("The Perceptron Learning Algorithm is one of the simplest algorithms in machine learning, used for binary classification problems. It was introduced by Frank Rosenblatt in 1958 and is the foundation of many modern neural network algorithms."
             )
    st.markdown("""<h4><b>Key Idea:</b></h4>
                    <p>The perceptron algorithm learns a linear decision boundary to separate two classes in a dataset. It adjusts the weights of the features iteratively to minimize classification errors.</p>
                <b><h4>Assumptions:</h4></b>
                <ul>
                    <li><b>Linearly Separable Data</b>:The perceptron assumes that the dataset is linearly separable. This means that there exists a hyperplane that can completely separate the two classes without any errors.</li>
                    <li> <b>Binary Classification</b>:The algorithm is designed specifically for binary classification problems.</li>
                    <li><b>Fixed Learning Rate:</b>The perceptron algorithm assumes that the learning rate is fixed and does not change during training.</li>
                </ul>
                """,unsafe_allow_html=True)
    st.markdown("""<h4><b>Working:</b></h4>Perceptron Algorithm is an early machine learning algorithm 
                that seeks to find a <b>linear decision boundary(a hyperplane in n dimensions)</b> that separates two classes in a dataset.it updates
                its perameters based on the errors made by incorrectly classified points.In geometric terms it moves the decision boundary towards misclassified
                points to correct the error.""",unsafe_allow_html=True)
    st.markdown("""
        <h5><b>Equation of Linear Decision Boundary(Hyper plane):<b></h5>
            The equation of a hyperplane in n-dimensional space is a generalization of the equation of a line (in 2D) or a plane (in 3D). In mathematical terms, a hyperplane in n-dimensions can be expressed as:
            <div style="font-size: 16px; font-family: Arial, sans-serif;display:flex;flex-direction:row;align-items:center;justify-content:center;">
                w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + &hellip; + w<sub>n</sub>x<sub>n</sub> + b = 0
            </div>
            <div>
                Where:
                <ul>
                    <li>x1,x2,x3,.....,xn are the input features in n-dimensional space</li>
                    <li>w1,w2,w3,......,wn are the corresponding weight of input features</li>
                    <li>b is the bias term(intercept)</li>
                </ul>
                In matrix Algebra this is the linear combination of the Weight vector(<b>W<sup>T</sup></b>) and the input Feature Vector(<b>X</b>).where 
                <ul>
                    <li>W=[w<sub>0</sub>,w<sub>1</sub>,w<sub>2</sub>,..,w<sub>n</sub>] is the weight vector</li>
                    <li>X=[x<sub>0</sub>,x<sub>1</sub>,x<sub>2</sub>,..,x<sub>n</sub>] is the input feature vector</li>
                </ul>  
                This can also be written as the dot product between the weight vector and the input feature vector i.e W<sup>T</sup>X=0.  
            </div>
    """,unsafe_allow_html=True)
    st.subheader("Labeling the Classes")
    st.markdown("""
        <div>
            <ul>
                <li>
                    In a binary classification problem, the classes are typically labeled as 0 and 1 (or +1 and -1 for some implementations).
                </li>
                <li>
                    The perceptron algorithm assumes that the training data is labeled with these values. A data point is assigned a label y such that:
                    <div style="font-size: 16px; font-family: Arial, sans-serif;">
                    y &in; {0, 1}
                    </div>
                    <div style="font-size: 16px; font-family: Arial, sans-serif;">
                    y &in; {-1, +1}
                    </div>
                </li>
                <li>
                    The positive region is where the weighted sum of the input, W<sup>T</sup>X > 0 where W=[w0,w1,w2,...,wn] and X=[1,x1,x2,...,xn].<br>In this region,the perceptron algorithm classifies the data point as belonging to Class 1 or positive class. 
                </li>
                <li>
                    The negative region is where the weighted sum of the input,W<sup>T</sup>X < 0 where W=[w0,w1,w2,...,wn] and X=[1,x1,x2,...,xn].<br>In this region, the perceptron algorithm classifies the data point as belonging to Class 0 or negative class.
                </li>
            </ul>
        </div>
    """,unsafe_allow_html=True)
    st.markdown(
    """
    In case of 2D space, the equation of the hyperplane is given by:
    \\[
    w_1 x + w_2 y + w_0 \\cdot 1 = 0
    \\]
    where \\( W = [w_0, w_1, w_2] \\) and \\( X = [1, x, y] \\).
    """, 
    unsafe_allow_html=True
    )

    st.image("regions.jpg", caption='Labeling classes')

    st.markdown(
        """
        In the above picture:
        - The **red line** represents the equation \\( 2x + 3y + 2 = 0 \\).
        - The **blue region** corresponds to the positive region (\\( y = +1 \\)), where any point satisfies the equation \\( 2x + 3y + 2 > 0 \\).
        - The **green region** corresponds to the negative region (\\( y = -1 \\)), where any point satisfies the equation \\( 2x + 3y + 2 < 0 \\).
        """
    )
    st.subheader("Algorithm:")
    st.markdown("""
        <ul>
            <li>
                <b>Step1</b>:Start with a random line with positive and negative regions
            </li>
            <li>
                <b>Step2</b>:Pick a large number(number of repetitions,or epochs) ex:1000
            </li>
            <li>
                <b>Step3</b>:(repeat 1000 times)
                <ul>
                    <li>pick a random point</li>
                    <li>if it is correctly classified:
                        <ul>
                            <li><b>Do Nothing</b></li>
                        </ul>
                    </li>
                    <li>
                        If point is incorrectly classified then
                        <ul>
                            <li><b>Move the line towards the point.</b></li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li>Enjoy the line that seperates the data.</li>
        </ul>
    """,unsafe_allow_html=True)
    st.subheader("How to move the line towards the point??")
    st.write("There are 2 ways to move a line.")
    st.markdown("""
        <ol>
            <li><b>Translation</b></li>
            <li><b>Rotation</b></li>
        </ol>
    """,unsafe_allow_html=True)
    st.markdown("""
        <h5>
            1)Translation or Change in Intercept (c)
        </h5>
        <ol>
            <li>If c increases(+):The line shifts upward without changing its slope.</li>
            <li>If c decreases(-):The line shifts downward without changing its slope.</li>
        </ol>
    """,unsafe_allow_html=True)
    st.image("Translation.jpg",caption="Translation of line by changing the C value", use_column_width=True)
    st.markdown("""
        <h5>
            2)Rotating(Changing the coefficient of X):
        </h5>
        <ol>
            <li>If a increases(+):The line rotates about y axis in clock wise direction</li>
            <li>If a decreases(-):The line rotates about y axis in counter clock wise direction</li>
        </ol>
    """,unsafe_allow_html=True)
    st.image("rotation1.jpg",caption="rotating about y axis",use_column_width=True)
    st.markdown("""
        <h5>
            3)Rotating(Changing the coefficient of y):
        </h5>
        <ol>
            <li>If a increases(+):The line rotates about x axis in counter clock wise direction</li>
            <li>If a decreases(-):The line rotates about x axis in clock wise direction</li>
        </ol>
    """,unsafe_allow_html=True)
    st.image("rotation2.jpg",caption="rotating about y axis",use_column_width=True)
    #positive region
    st.subheader("Moving a line towards the point in positive region")
    st.write("""To move the line towards the point(p,q) in the positive region we need to do 
    2 rotations(small decrease the coefficients of x and y) and 1 translation(small decrease the c).here the small decreament is given by learning rate&eta;..usually it is takes as 0.1,0.01,0.001 etc..""")
    st.image("p1.jpg",caption="point in positive region of line")
    st.image("p2.jpg",caption="decrementing a by 0.1*p")
    st.image("p3.jpg",caption="decrementing a by 0.1*p and b by 0.1*q")
    st.image("p4.jpg",caption="decrementing a by 0.1*p and b by 0.1*q and c by 0.1")
    #negative region
    st.subheader("Moving a line towards the point in negative region")
    st.write("""To move the line towards the point(p,q) in the negative region we need to do 
    2 rotations(small increase the coefficients of x and y) and 1 translation(small increase the c).here the small increment is given by &eta;*coefficient..usually it is takes as 0.1,0.01,0.001 etc..""")
    st.image("n1.jpg",caption="point in negative region of line")
    st.image("n2.jpg",caption="incrementing a by 0.1*p")
    st.image("n3.jpg",caption="incrementing a by 0.1*p and b by 0.1*q")
    st.image("n4.jpg",caption="incrementing a by 0.1*p and b by 0.1*q and c by 0.1s")
    #modified algorithm
    col1,col2=st.columns(2)
    col1.markdown("""
        <h5>Modified Algorithm:</h5>
        <ul>
            <li>
                <b>Step1</b>:Start with a random line of equation ax+by+c=0
            </li>
            <li>
                <b>Step2</b>:Pick a large number(number of repetitions,or epochs) ex:1000
            </li>
            <li><b>Step3: </b>pick a small number.0.01(learning rate)</li>
            <li>
                <b>Step4</b>:(repeat 1000 times)
                <ul>
                    <li>pick a random point (p,q)</li>
                    <li>if it is correctly classified:
                        <ul>
                            <li>Do Nothing</li>
                        </ul>
                    </li>
                    <li>
                        If negative point is present in the positive region.
                        <ul>
                            <li>subtract &eta;*p to a</li>
                            <li>subtract &eta;*q to b</li>
                            <li>subtract &eta; to c</li>
                        </ul>
                    </li>
                    <li>
                        If positive point is present in the negative region.
                        <ul>
                            <li>add &eta;*p to a</li>
                            <li>add &eta;*q to b</li>
                            <li>add &eta; to c</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li>Enjoy the line that seperates the data.</li>
        </ul>
    """,unsafe_allow_html=True)
    col2.markdown("""
        <h5>
            Pseudocode:
        </h5>
        """,unsafe_allow_html=True)
    with col2.container():
        col2.code("""
            epochs = 1000
            learning_rate = 0.01
            w = [n+1 random values, usually 1's]
            for i in range(epochs):
                #consider a random point
                if x is negative point and (W^T)*x>0:
                w-=learning_rate*x
                if x is positive point and (W^T)*x<0:
                w+=learning_rate*x
            """, language='python')
    with col2.container():
        col2.markdown("""<h4>Simplified Algorithm: </h4>""",unsafe_allow_html=True)
        col2.code("""
            epochs = 1000
            learning_rate = 0.01
            w = [n+1 random values, usually 1's]
            for i in range(epochs):
                #consider a random point
                w=w+learning_rate*(y-y^)*x
        """)
    with col2.container():
        col2.markdown("here n is the no.of input features and W is the weight vector and x is the input feature vector.y is the original value of the output label and y^ is the predicted value and is given by ŷ=step(z)where z=W<sup>T</sup>X",unsafe_allow_html=True)
    st.subheader("Vizualization of perceptron learning Algorithm: ")
    st.image("perceptronAnimation.gif")
    st.subheader("Drawbacks of Perceptron Learning Algorithm: ")
    st.markdown("""
            <ul>
                <li><b>Limited to Linearly Separable Data:</b>The perceptron can only solve problems where the classes are linearly separable. If the data is not linearly separable, the perceptron will not converge, meaning it will not find a solution that separates the classes correctly.</li>
                <li><b>No Probabilistic Interpretation:</b>The output of the perceptron is binary (0 or 1), without any measure of confidence. This lack of probabilistic output makes it difficult to interpret the output as a probability score, which is often required in applications where uncertainty needs to be considered.</li>
                <li><b>No Capability for Complex Patterns:</b>The basic perceptron can only represent linear decision boundaries. It cannot learn complex patterns that require non-linear decision surfaces.</li>
                <li><b>Convergence Issues:</b>The algorithm will only converge if the data is linearly separable. For non-linearly separable data, the learning algorithm may not find a solution, leading to an infinite loop or failure to reach a convergence point.</li>    
            </ul>
        """,unsafe_allow_html=True)
    st.markdown("""
    <style>
        a,a:hover{
                text-decoration:none;
        }
    </style>
    <div class="resources-section">
        <h3>Resources</h3>
        <ul>
            <li><a href="https://github.com/TirumalaRaoBoddana/BinaryClassification/blob/main/perceptron.py" target="_blank">Code Implementation</a></li>
            <li><a href="https://youtube.com/playlist?list=PLKnIA16_Rmvb-ZTsM1QS-tlwmlkeGSnru&si=-ZN0OwDBkT8pAWgr" target="_blank">CampusX logistic regression playlist</a></li>
            <li><a href="https://youtu.be/jbluHIgBmBo?si=jfxIBtymPODwvLyf" target="_blank">Binary classification using Perceptron by Seranno.academy</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
