# streamlit run app.py
import streamlit as st 
from streamlit_agraph import agraph, Node, Edge, Config

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def get_decision_tree_code() :
    """ Retourne le code associé à au model """

    return f"""from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Data
X = data[{feature_columns}]
y = data['{target_column}']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# Create the model
model = DecisionTreeClassifier(
    criterion='{criterion}',
    splitter='{splitter}',
    max_depth={max_depth},
    min_samples_split={min_samples_split}
)
model.fit(X_train, y_train)"""

def generate_n_colors(n):
    """
    Generate n distinct colors.

    Parameters:
    n (int): The number of distinct colors to generate.

    Returns:
    List[str]: A list of n colors in hex format.
    """
    colors = plt.cm.hsv(np.linspace(0, 1, n+1))
    return [mcolors.rgb2hex(color) for color in colors][1:]

def percent_kpi_chart(percent : float, title : str, color : str = '#00a67d') :
    """ Retourne un KPI façon pourcentage (basé sur un pie plot vega lite) """
    df = pd.DataFrame({'value': [percent]})
    
    text = f"{percent:.1%}"
    radius = {"innerRadius":100,"outerRadius":120}

    return st.vega_lite_chart(df, {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title" : {"align": "left", "anchor": "start", "text" : f"{title} : {percent:.1%}"},
        "layer": [
            {
                "mark": {
                    "type": "arc", "theta":-1.7, 
                    "theta2":percent*2*3.14-1.7,
                    "color" : f"{color}", 
                    "cornerRadius":15 if percent < 1 else 0,
                    **radius
                }
            },
            {
                "mark":{
                    "type": "text", "fontSize":42, 
                    "text": text,
                }
            }   
        ],
        # 'width' : 350,
        "config" : {"mark": {"tooltip": False}}
    },
        theme=None, use_container_width=True
    )

def plot_decision_tree_graph(model : DecisionTreeClassifier, feature_names):
    nodes = []
    edges = []
    values = model.tree_.value
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    colors = generate_n_colors(len(model.classes_))

    def recurse(node, parent=None, depth=0):
        if feature[node] != -2:  # -2 indicates a leaf node
            node_label = (f"{feature_names[feature[node]]} <= {threshold[node]:.2f}")
            color = "#ababab"
        else:
            node_label = f"{model.classes_[values[node].argmax()]}"
            color = colors[values[node].argmax()]

        nodes.append(Node(id=str(node), label=node_label, color=color))

        if parent is not None:
            edges.append(Edge(source=str(parent), target=str(node)))

        if feature[node] != -2:  # -2 indicates a leaf node
            left_child = model.tree_.children_left[node]
            right_child = model.tree_.children_right[node]
            recurse(left_child, node, depth + 1)
            recurse(right_child, node, depth + 1)

    recurse(0)

    opts = {
        "layout": {
            "hierarchical": {
                "direction": "UD",
                "sortMethod": "directed",
            },
        },
        "edges": {
            "arrows": "to",
        }
    }
    config = Config(width=710, height=710, directed=True, physics=False, hierarchical=True, **opts)
    return agraph(nodes=nodes, edges=edges, config=config)

@st.experimental_dialog("Preview", width='large')
def data_preview(df : pd.DataFrame):
    if st.toggle('Head', True) :
        df = df.head(10)
    st.dataframe(df)

# App title
st.set_page_config(page_title='Decision Tree Builder', page_icon="🌳", layout='centered')
st.logo('https://www.luc-estienne.com/web/image/website/1/logo', link='https://www.luc-estienne.com/')

st.header('🌳 Decision Tree Builder', divider='gray')
"""
Discover an interactive interface for creating and visualising classification models with decision trees. 
Import your CSV data, adjust the parameters, view the tree and evaluate the performance of your models intuitively and efficiently.
"""

# --------- UI
form_side = st.container()
main_side = st.container()

uploaded_file = form_side.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    accuracy, ineractive_graph, static_image, tree_code, prediction = main_side.tabs(['Accuracy', 'Interactive Graph', 'Satic Image', 'Python Code', 'Prediction'])
    data = pd.read_csv(uploaded_file)

    # SIDEBAR
    with st.sidebar :
        "## Hyperparameters"

        # Sélection des colonnes
        target_column = st.selectbox("Select Target Column", data.columns, help='The target column to predict.')
    
        initial_feature_columns = [col for col in data.columns if col != target_column]
        feature_columns = initial_feature_columns
        if st.toggle('Select Feature Columns', False, help='The feature columns to use for training the model.') :
            feature_columns = st.multiselect("Select Feature Columns", initial_feature_columns)

        # Hyperparameters
        with st.expander('More settings', icon=':material/settings:') : 
            _col1, _col2 = st.columns(2)
            criterion = _col1.selectbox("Criterion", ["gini", "entropy", "log_loss"], help='The function to measure the quality of a split')
            splitter = _col2.selectbox("Splitter", ["best", "random"], help='The function to measure the quality of a split')
            
            max_depth = None
            if st.toggle('Maximum depth', True, help='The maximum depth of the tree') : 
                max_depth = st.slider("Max Depth", 1, 20, value=4)
            min_samples_split = st.columns(2)[0].number_input("Min Samples Split", 2, None, help='The minimum number of samples required to split an internal node')

            # '---'
            test_size = st.slider("Test Size", 0.1, 0.5, value=0.2, help="The proportion of the dataset to include in the test split.")

            random_state = None
            if not st.toggle('Random seed', True, help='The seed used by the random number generator.') :
                random_state = st.number_input("Random State", 1, None, 42)

    # auto_launch = form_side.toggle('Auto launch', False)

    _col1, _col2, _col3 = form_side.columns(3)
    if _col1.button('Show data', use_container_width=True) : data_preview(data)

    if _col2.button("Train Model", type='primary', use_container_width=True):
        if not target_column :
            st.warning("Please select the target column.", icon=':material/error:')
        if not feature_columns :
            st.warning("Please select at least one feature column.", icon=':material/error:')

        if target_column and feature_columns:
            with st.spinner('Training...') :
                X = data[feature_columns]
                y = data[target_column]

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Create the model
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    splitter=splitter,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                model.fit(X_train, y_train)

                # Predict and evaluate
                st.session_state.val_accuracy = accuracy_score(y_test, model.predict(X_test))
                st.session_state.train_accuracy = accuracy_score(y_train, model.predict(X_train))
                st.session_state.model = model

    if 'model' in st.session_state.keys() :
        if _col3.button("Show Tree", use_container_width=True):
            with main_side :
                with st.spinner('Generation...') :
                    # Plot the tree
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(st.session_state.model, feature_names=feature_columns, class_names=True, filled=True, ax=ax)
                    static_image.pyplot(fig)

                    with ineractive_graph.container(border=True) :
                        plot_decision_tree_graph(st.session_state.model, feature_names=feature_columns)

        with prediction :
            _col1, _col2 = st.columns(2)
            with _col1 :
                '#### Insert data :'
                df = st.data_editor(pd.DataFrame(0, index=[0], columns=feature_columns), hide_index=True)
                y_proba = st.session_state.model.predict(df)

            with _col2 :
                '#### Prediction :'
                f'`{y_proba[0]}`'

    if 'accuracy' in st.session_state.keys() :
        with accuracy :
            _col1, _col2 = st.columns(2)
            with _col1 : percent_kpi_chart(st.session_state.train_accuracy, 'Training Accuracy')
            with _col2 : percent_kpi_chart(st.session_state.val_accuracy, 'Test Accuracy')

    tree_code.code(get_decision_tree_code())