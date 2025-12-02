import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import streamlit as st
import base64

### Import trained logistic regression model + functions
from Avalanche_Logistic_Regression import (
    sigmoid,
    train_log_reg
)

@st.cache_resource  # caches across reruns so you don't retrain every slider move
def load_and_fit_models(n_boot=100, n_steps=1000, step_size=0.01):
    # Read CSVs
    train_df = pd.read_csv("train_avalanche_data.csv")
    test_df  = pd.read_csv("test_avalanche_data.csv")

    # First 20 columns are features, column 21 is labels
    X_train_raw = train_df.iloc[:, :20].values
    y_train     = train_df.iloc[:, 20].values

    X_test_raw  = test_df.iloc[:, :20].values
    y_test      = test_df.iloc[:, 20].values

    # Add bias column
    N_train = X_train_raw.shape[0]
    N_test  = X_test_raw.shape[0]

    X_train = np.hstack([np.ones((N_train, 1)), X_train_raw])
    X_test  = np.hstack([np.ones((N_test, 1)),  X_test_raw])

    # Single logistic regression fit (MLE-ish)
    w_hat = train_log_reg(X_train, y_train, n_steps=n_steps, step_size=step_size)

    # Bootstrapping to get weight distribution
    boot_weights = np.zeros((n_boot, X_train.shape[1]))

    # Add progress bar for Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()

    for b in range(n_boot):
        idx = np.random.randint(0, N_train, size=N_train)
        Xb = X_train[idx]
        yb = y_train[idx]
        wb = train_log_reg(Xb, yb, n_steps=n_steps, step_size=step_size)
        boot_weights[b, :] = wb

        # Update progress
        if b % 10 == 0:
            progress_bar.progress((b + 1) / n_boot)
            status_text.text(f"Bootstrapping: {b+1}/{n_boot}")
    progress_bar.empty()
    status_text.empty()

    # Also compute test accuracy once (for display)
    p_test = sigmoid(X_test @ w_hat)
    y_pred = (p_test >= 0.5).astype(int)
    test_acc = (y_pred == y_test).mean()

    return w_hat, boot_weights, test_acc

### General streamlit UI/organization
# Background image
with open("Rainier.webp", "rb") as f:
        data = f.read()
encoded = base64.b64encode(data).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True)
# Title
st.markdown(
    """
    <div style = 'text-align: center;
                font-size: 75px; 
                font-weight: bold; 
                font-family: Helvetica, sans-serif; 
                color: #fcfeff;'>
        Uncertain Slopes
    </div>
    """,
    unsafe_allow_html=True
)

# Add organizing pages
tab_cover, tab_motivation, tab_background, tab_methods, tab_model, tab_references = st.tabs([
    "Project Cover", "Project Motivation", "Avalanche Background", "Project Methods",
    "Avalanche Model", "References"
])

with tab_cover:
    st.write("""
                **Uncertain Slopes** is a submission to Stanford's CS109 final project challenge for Fall 2025. Its goal is to create a more transparent distribution of a particular slope's
             avalanche probability (ie, an uncertain slope!), rather than a single point value.
                 """)

with tab_motivation:
    st.image("deep-persistent-slab.webp", "Avalanche forecasters assess a deep persistant slab in the Cascades.")
    st.subheader("Why this topic", )
    st.write("Avalanche danger is not an important concern to the vast majority of people. For those interested in snow sports that take place away from ski resorts, like backcountry skiing, snowmobiling, or mountaineering, avalanches can be a death sentence. It takes a mere 18 inches of snow to trap and kill a fully grown person, and Avalanches account for over 75% of backcountry ski fatalities in the United States. Of these avalanche fatalities, a specific type of avalanche called a **slab avalanche** accounts for 90% of deaths. This project aims to predict slab avalanches.")
    st.image("Slab-Loose.png", "Loose snow avalanches vs. slab avalanches")
    st.write("I've skied since I was four, and I started backcountry skiing in Tahoe a couple years ago. The danger of avalanches have become pretty evident to me over the past few years. With all these risks, it's super important to have accurate avalanche forecasting. Current avalanche education through the American Institute for Avalanche Research and Education (AIARE) classes emphasizes the North American Avalanche Danger Scale to assess risk. Such a scale is often found online for a local area.")
    st.image("North_American_Public_Danger_Scale_Icons.jpg", caption="The North American Avalanche Danger Scale", width = 500)
    st.write("However, a big problem I've found when using this model is transparency. Simply sorting ratings into one of five categories (Low, Moderate, Considerable, High, Extreme) hides how uncertain the underlying conditions actually are. Two “Moderate” days might have wildly different risks - one might mean a 5% chance of triggering a slide, another might mean 25% - yet they both collapse into the same bucket. The scale gives a single categorical label, but it doesn't communicate how confident forecasters are. The goal of Uncertain Slopes is to output a probability distribution that communicates confidence of an avalanche occurring on a given slope, based on 20 principle features (sourced from the AIARE 1 guidebook). For a list/definition of these terms, see the Avalanche Background page")

with tab_background:
    st.header("What Is An Avalanche?")
    st.image("Avalanche-Problems-Overview-1.jpg", "Different types of avalanche problems")
    st.write("An avalanche is a rapid flow of snow down a mountainside caused when the snowpack can no longer support its own weight. Most dangerous avalanches are slab avalanches, where a cohesive sheet of snow suddenly breaks away along a weak layer buried below the surface. These slabs can accelerate to highway speeds within seconds, easily overwhelming skiers, climbers, and snowmobilers. The structure of the snowpack, recent weather, terrain features, and temperature all combine to determine how likely a slope is to slide on any given day.")
    
    st.header("Principle Features of Prediction")

    st.subheader("Surface/Depth Hoar")
    st.image("surface-hoar-2.jpg", "Surface Hoar")
    st.write("Surface hoar and depth hoar are weak, sugary snow crystals that form when the snowpack becomes very cold and faceted. These layers bond poorly to the surrounding snow, creating “persistent weak layers” that can fail long after a storm. Avalanches triggered on hoar layers are often large and destructive.")
    
    st.subheader("Slope Angle")
    st.image("f128a66e3390cc74ff1e451404ad4ce7408a60f6_unawares_slopeangle_offset.avif", "Critical Avalanche Slope Angles", width = 750)
    st.write("Most fatal slab avalanches occur on slopes between 30° and 45°, the sweet spot where snow is steep enough to slide but not so steep that it sluffs off naturally. Avalanches are basically impossible at angles <30 degrees.")
   
    st.subheader("Slope Convexity")
    st.image("convex.avif", "Convexity in Avalanches")
    st.write("Convex (rollover) slopes bulge outward, creating tension within the snowpack. These tension zones are common trigger points because the snowpack is being pulled apart.")
    
    st.subheader("Anchored Terrain")
    st.image("Anchors.jpeg", "Anchors in Avalanches")
    st.write("Anchors like trees, rocks, and bushes help stabilize the snowpack by physically holding layers together. Slopes with abundant anchors are generally safer. Slab avalanches are basically impossible below treeline.")
    
    st.subheader("Current Temperature")
    st.image("Snow-Metamorphism_temp-gradient-Custom.webp", "Temperature Gradient in Snowpack")
    st.write("Rapid warming weakens bonds between snow crystals and can quickly destabilize the snowpack. Warm temperatures also promote the formation of wet loose avalanches and weaken slab structures.")
    
    st.subheader("Sun Exposure")
    st.image("img_8275-3.jpg", "Two Different Aspects")
    st.write("Slopes facing the sun warm up faster, making the snowpack more prone to melting, crust formation, and mid-day instability.")
    
    st.subheader("Nearby Avalanches")
    st.write("Avalanches that occurred in the past 24-48 hours are one of the strongest indicators that the snowpack remains unstable. They show that similar slopes with similar exposure are likely to fail.")
    
    st.subheader("Recent Snowfall")
    st.write("Large amounts of new snow (especially >30 cm/12 inches in 24 hours) add weight to the snowpack and can overload weak layers.")
    
    st.subheader("Rain on Snow")
    st.write("Rain percolates down into the snowpack and destroys bonding between crystals, creating extremely unstable conditions.")
    
    st.subheader("Days Since Storm")
    st.write("Hazard typically decreases the longer it has been since the last storm because the snowpack gradually settles and strengthens.")

with tab_methods:
    st.header("How was this project made?")

    st.subheader("Overview")
    st.write(
        """
        This project is a simple avalanche
        forecasting tool that tries to tell you **how likely** a slab avalanche
        is on a certain day in the backcountry. Instead of just saying
        "Low" or "High" danger, it tries to give a **full probability
        distribution** for how risky that day might be. To do this, I generated a synthetic dataset, then trained logistic regression, bootstrapped a distribution of weights, then outputted that distribution as a probability.
        """)
    st.subheader("Generating the dataset")
    st.write("""
        To build the dataset, I needed to generate a bunch of synthetic data (450 train entries, and 50 test entries). To accurately generate the 20 features for each day, I generate four class paradigms that described
        the weather for a given day: STORM, COLD_CLEAR, WARM_WET, DRY_PERSISTANT with probabilities 0.15, 0.25, 0.20, and 0.40. Within each of these paradigms, I defined the joint
        probabilities of each feature in the table below.""")
    data = {
    "Feature": [
        "Surface Hoar",
        "Depth Hoar",
        "Slope >35 Degrees",
        "Concave Slope",
        "Anchored Terrain",
        "Sun Crust",
        "Warmer Air Temp",
        "Colder Air Temp",
        "Avalanches Nearby",
        "Strong Winds",
        "Weak Layer Within 60cm",
        "Rain on Snow last 24 hrs",
        "Rapid Warming",
        "Sun Exposure",
        "Above Treeline",
        "Early Season",
        "Overnight Refreeze",
        "3 days since last storm",
    ],
    "P(Feature|STORM)": [
        0.05, 0.10, 0.60, 0.30, 0.40, 0.10, 0.30, 0.10, 0.25, 0.70,
        0.40, 0.10, 0.10, 0.30, 0.50, 0.40, 0.20, 0.10
    ],
    "P(Feature|COLD_CLEAR)": [
        0.30, 0.15, 0.60, 0.30, 0.40, 0.10, 0.10, 0.80, 0.05, 0.20,
        0.35, 0.00, 0.00, 0.70, 0.50, 0.25, 0.70, 0.70
    ],
    "P(Feature|WARM_WET)": [
        0.05, 0.05, 0.60, 0.30, 0.40, 0.10, 0.80, 0.05, 0.15, 0.40,
        0.30, 0.40, 0.60, 0.80, 0.50, 0.15, 0.20, 0.50
    ],
    "P(Feature|DRY_PERSISTANT)": [
        0.25, 0.25, 0.60, 0.30, 0.40, 0.10, 0.40, 0.20, 0.15, 0.30,
        0.50, 0.05, 0.10, 0.60, 0.50, 0.20, 0.30, 0.80
    ],
    }

    df = pd.DataFrame(data)
    df.round(0)
    st.table(df)

    st.write("""
        Generating the labels was a bit trickier, and was the weakest link of my project. I ended up going with a 'red-flag' + 'requirements' algorithm that hard-coded the requirements
        for an avalanche (like above treeline + slope angle > 30 degrees) and red flags like depth hoar, warming, etc. Then I calculated the labels as the output of a Bernoulli with
        p decided by the algorithm below. 
        """)
    st.image("red-flag.jpg", "red-flag algorithm")
    
    
    st.subheader("Logistic Regression")
    st.write("""
    To train a logistic regression model on the dataset, I used the classic MLE gradient ascent method we learned in class. I also allocated some of the synthetic dataset
    to test the model's accuracy, which I show in the avalanche_model tab.
    """)
    st.image("log_reg.jpg", "Code for logistic regression")

    st.subheader("Bootstrapping")
    st.write("""
    To find a distribution of the probability of an avalanche, I first found a distribution of the weights by running logistic regression 300 times on the synthetic dataset. I 
    then converted this weights distribution to a probability distribution using sigmoid, and fitted a Gaussian to the data to show a smooth curve. Finally, I outputted the MLE estimate
     to show how the distribution includes confidence.
    """)
    st.image("Bootstrap.jpg", "Bootstrapped probability distribution")

with tab_model:
    st.title("Avalanche Risk Forecasting")

    with st.spinner("Wait as we train the logistic regression model..."):
        w_hat, boot_weights, test_acc = load_and_fit_models()

    st.success(f"Model trained! Test accuracy: {test_acc:.3f}")

    st.markdown("---")

    st.header("Test a Specific Day's Conditions")

    st.write("Toggle the 20 binary features below. The **default** is your current example day.")

    # Default inputs
    default_x_star_raw = np.array([
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0
    ])

    feature_names = [
        "Surface hoar present",
        "Depth hoar present",
        "Average slope > 30°",
        "Average slope > 40°",
        "Concave slope",
        "Anchored terrain",
        "Sun crust",
        "Warmer day",
        "Colder day",
        "Avalanches nearby",
        ">30cm of snowfall in last 24 hrs",
        "Strong winds >35 mph",
        "Weak layer <60 cm deep",
        "Recent rain on snow",
        "Rapid warming",
        "Heavy sun exposure",
        "Above treeline",
        "Early season",
        "Overnight refreeze",
        ">3 days since last storm",
    ]

    # Sidebar controls for all features
    st.subheader("Today's Conditions")
    user_features = []
    cols = st.columns(2)

    for i, name in enumerate(feature_names):
        default_val = bool(default_x_star_raw[i])
        col = cols[i % 2]  # send them alternately to col 0 / col 1
        with col:
            val = st.checkbox(
                name,
                value=default_val,
                key=f"feat_{i}"  # unique key so Streamlit doesn't get confused
            )
        user_features.append(int(val))

    x_star_raw = np.array(user_features, dtype=float)
    x_star = np.concatenate(([1.0], x_star_raw))  # add bias

    ### Once we get input compute bootstrap distribution and display
    this_theta = boot_weights @ x_star   # shape (n_boot,)
    p_boot = sigmoid(this_theta)

    mu = p_boot.mean()
    sigma = p_boot.std()

    # Also compute MLE p using w_hat
    p_mle = sigmoid(w_hat @ x_star)

    st.subheader("Avalanche Probability for This Day")

    col1, col2, col3 = st.columns(3)
    col1.metric("Bootstrap mean p", f"{mu:.1%}")
    col2.metric("Bootstrap std", f"{sigma:.1%}")
    col3.metric("MLE p (single fit)", f"{p_mle:.1%}")

    st.caption(
        "Mean and std are computed from the bootstrap distribution of probabilities; "
        "the dashed line in the plot below is the single logistic-regression fit (no bootstrapping)."
    )

    ### Plot histogram:
    fig, ax = plt.subplots()

    ax.hist(p_boot, bins=20, range=(0, 1), density=True, alpha=0.4, label="Bootstrap histogram")

    if sigma > 0:
        xs = np.linspace(0, 1, 400)
        gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        ax.plot(xs, gauss, linewidth=2, label="Gaussian fit")

    ax.axvline(p_mle, linestyle="--", linewidth=2, label=f"MLE p = {p_mle:.2f}")

    ax.set_xlabel("Avalanche probability p")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap distribution of avalanche probability for chosen day")
    ax.set_xlim(0, 1)
    ax.legend()

    st.pyplot(fig)

with tab_references:
    st.subheader("References")
    st.write("Here are the references I used for this project:")

    st.write(
        """

        *“SLABS: An Improved Probabilistic Method to Assess the Avalanche Risk on Backcountry Ski Tours.”* 
        *ScienceDirect*, 2024, https://www.sciencedirect.com/science/article/abs/pii/S0165232X24002921.

        *“Avalanche Types.”* SLF Institute for Snow and Avalanche Research, 
        https://www.slf.ch/en/avalanches/avalanche-science-and-prevention/avalanche-types/.

        *AIARE 1 Student Manual: 2024 - 25.* American Institute for Avalanche Research and Education, 2024, 
        https://avtraining.org/wp-content/uploads/2024/11/AIARE-1-Student-Manual_2024%E2%80%9325.pdf.

        *“Deep Persistent Slabs: Understanding a Tricky Avalanche Problem.”* Utah Avalanche Center, 2024, 
        https://utahavalanchecenter.org/blog/67676.

        As I don't have experience with Numpy and Pandas, I also used ClaudeCode to help rewrite my logistic regression/bootstrapping code
        and speed it up with vectorization
        """
    )
