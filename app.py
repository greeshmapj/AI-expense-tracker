
import os
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pytesseract
import cv2
import csv
import numpy as np
import joblib
import re
from PIL import Image
from sklearn.linear_model import LinearRegression
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

#Page Layout
st.set_page_config(page_title="AI Expense Tracker",page_icon="💰",layout="wide")

# LOAD CSS
with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

#Create Save Function
def save_expense(date, description, amount, category, mode):
    new_data = pd.DataFrame({'Date': [pd.to_datetime(date).strftime('%Y-%m-%d')],'Description': [description],'Amount': [amount],'Category': [category],'Input Mode': [mode]})
    new_data.to_csv('expenses.csv',mode='a',header=not os.path.exists('expenses.csv'),index=False,quoting=csv.QUOTE_ALL)

# PDF REPORT FUNCTION
def generate_pdf_report(df, selected_month):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph(
        f"AI Expense Tracker Report - {selected_month}",
        styles['Title']
    )
    elements.append(title)
    elements.append(Spacer(1, 20))

    # Summary
    total_expense = df['Amount'].sum()
    highest_expense = df['Amount'].max()
    summary = Paragraph(
        f"""
        <b>Total Expenses:</b> ₹ {total_expense:.2f}<br/>
        <b>Highest Expense:</b> ₹ {highest_expense:.2f}<br/>
        <b>Total Transactions:</b> {len(df)}
        """,
        styles['BodyText']
    )
    elements.append(summary)
    elements.append(Spacer(1, 20))

    # Table Data
    table_data = [['Date', 'Description', 'Amount', 'Category']]

    for _, row in df.iterrows():
        table_data.append([
            str(row['Date']),
            str(row['Description'])[:30],
            f"₹ {row['Amount']}",
            str(row['Category'])
        ])

    # Create Table
    table = Table(table_data)

    # Style Table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
    ]))
    elements.append(table)

    # Build PDF
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

#OCR
# Tesseract Path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Model
model = joblib.load('expense_classifier.pkl')

# =========================
# SIDEBAR NAVIGATION
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "📊 Dashboard",
        "🧾 Receipt OCR",
        "✍ Manual Entry",
        "🎤 Voice Entry",
        "📜 Expense History"
        ]
)

if menu == "🧾 Receipt OCR":
    st.header("Receipt OCR")
    uploaded_file = st.file_uploader("Upload Receipt Image",type=['png', 'jpg', 'jpeg'])

    #OCR Extraction
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Receipt')
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()),dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        # Normalize OCR Text
        text = text.lower()
        text = ' '.join(text.split())

        #Show OCR Text
        st.subheader("Extracted Text")
        st.write(text)

        #Category Prediction
        prediction = model.predict([text])
        st.subheader("Predicted Category")
        edited_category = st.selectbox(
        "Confirm Category",
        [
            "Food",
            "Travel",
            "Bills",
            "Entertainment",
            "Healthcare",
            "Transport",
            "Grocery",
            "Clothing",
            "Electronics",
            "Online Shopping"
        ],
        index=[
            "Food",
            "Travel",
            "Bills",
            "Entertainment",
            "Healthcare",
            "Transport",
            "Grocery",
            "Clothing",
            "Electronics",
            "Online Shopping"
        ].index(prediction[0])
    )
        
        #Date Extraction
        date_match = re.search(r'(\d{2}/\d{2}/\d{2,4})', text)
        extracted_date = (date_match.group(1) if date_match else "Unknown")

        # Editable Date Field
        edited_date = st.text_input("Confirm Date", value=extracted_date)

        #Amount Extraction
        amount_patterns = [
        r'net amount\s*[:\-]?\s*(\d+[.,]\d{2})',
        r'grand total\s*[:\-]?\s*(\d+[.,]\d{2})',
        r'total amount\s*[:\-]?\s*(\d+[.,]\d{2})',
        r'amount paid\s*[:\-]?\s*(\d+[.,]\d{2})',
        r'total\s*[:\-]?\s*(\d+[.,]\d{2})'
    ]
        extracted_amount = 0

        for pattern in amount_patterns:
            match = re.search(pattern,text,re.IGNORECASE)
            if match:
                extracted_amount = float(match.group(1).replace(',', '.'))
                break

        # FALLBACK METHOD
        if extracted_amount == 0:
            amounts = re.findall(r'\d+[.,]\d{2}',text)
            if amounts:
                amounts = [float(a.replace(',', '.')) for a in amounts]
                extracted_amount = max(amounts)

        # Show Amount
        if extracted_amount > 0:
            st.write("Estimated Amount:",extracted_amount)
        else:
            st.write("Amount Not Found")

        # Editable Amount Field
        edited_amount = st.number_input("Confirm Amount",value=float(extracted_amount))

        #Save OCR Resoponse
        if st.button("Save Bill Expense"):
            save_expense(edited_date,text,edited_amount,edited_category, "Receipt OCR")
            st.success("Bill Expense Saved")

################################################

#MANUAL ENTRY SECTION
if menu == "✍ Manual Entry":
    st.header("Manual Expense Entry")

    #Add user input fields
    manual_description = st.text_input("Enter Expense Description")
    manual_amount = st.number_input("Enter Amount", min_value=0.0)
    manual_date = st.date_input("Select Date")

    #Add Predict Button
    if st.button("Predict & Save Manual Expense"):
        if manual_description:
            manual_prediction = model.predict([manual_description])

            st.success(f"Predicted Category: {manual_prediction[0]}")
            st.subheader("Expense Summary")
            st.write("Description:", manual_description)
            st.write("Amount:", manual_amount)
            st.write("Date:", manual_date)
            st.write("Category:", manual_prediction[0])

            save_expense(manual_date, manual_description, manual_amount, manual_prediction[0], "Manual")
            st.success("Expense Saved Successfully")

        else:
            st.warning("Please Enter Description")



##################################################

#VOICE RECOGNITION
# import speech_recognition as sr

# if menu == "🎤 Voice Entry":
#     st.header("Voice Expense Entry")
#     st.info("Speak continuously for up to 10 seconds")

#     # Initialize Session State
#     if "voice_text" not in st.session_state:
#         st.session_state.voice_text = ""
#     if "voice_prediction" not in st.session_state:
#         st.session_state.voice_prediction = "Food"
#     if "voice_amount" not in st.session_state:
#         st.session_state.voice_amount = 0.0

#     #Record Button
#     if st.button("Start Voice Recording"):
#         recognizer = sr.Recognizer()

#         try:
#             with sr.Microphone() as source:
#                 st.write("Listening...")
#                 audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
#                 voice_text = recognizer.recognize_google(audio)
#                 # Save Voice Text
#                 st.session_state.voice_text = voice_text

#                 # Extract Amount From Voice
#                 amount_match = re.search(r'(\d+[.,]?\d*)',voice_text)
#                 if amount_match:
#                     st.session_state.voice_amount = float(amount_match.group(1).replace(',', '.'))
#                 else:
#                     st.session_state.voice_amount = 0.0
            
#                 #Category Prediction
#                 prediction = model.predict([voice_text])
#                 st.session_state.voice_prediction = prediction[0]

#         except Exception as e:
#             st.error(f"Error: {e}")

#     #Show Extracted Voice Text
#     if st.session_state.voice_text != "":
#         st.subheader("Recognized Text")
#         st.write(st.session_state.voice_text)

#         # Editable Description
#         edited_voice_text = st.text_input("Confirm Description",value=st.session_state.voice_text)

#         # Editable Amount
#         edited_voice_amount = st.number_input("Confirm Amount",min_value=0.0,value=float(st.session_state.voice_amount))

#         # Categories
#         categories = [
#             "Food",
#             "Travel",
#             "Bills",
#             "Entertainment",
#             "Healthcare",
#             "Transport",
#             "Grocery",
#             "Clothing",
#             "Electronics",
#             "Online Shopping"
#         ]
#         # Editable Category
#         edited_voice_category = st.selectbox("Confirm Category",categories,index=categories.index(st.session_state.voice_prediction))

#         # Editable Date
#         edited_voice_date = st.date_input("Select Date",value=pd.Timestamp.today())

#         # SAVE BUTTON
#         if st.button("Save Voice Expense"):
#             save_expense(edited_voice_date,edited_voice_text,edited_voice_amount,edited_voice_category,"Voice")
#             st.success("Voice Expense Saved")
           
##################################################

# EXPENSE HISTORY
if menu == "📜 Expense History":
    st.header("Expense History")
    try:
        expense_df = pd.read_csv("expenses.csv")

        # Show newest first
        display_df = expense_df.iloc[::-1]
        st.dataframe(display_df,use_container_width=True)
        st.divider()

        # SELECT ROW
        st.subheader("Edit / Delete Expense")
        row_index = st.number_input("Enter Original Row Index",min_value=0,max_value=len(expense_df)-1,step=1)
        
        # Selected Row
        selected_row = expense_df.iloc[row_index]
        st.write("Selected Expense")
        st.dataframe(selected_row.to_frame().T,use_container_width=True)

        # EDIT FIELDS
        edited_date = st.text_input("Edit Date",value=str(selected_row['Date']))
        edited_description = st.text_input("Edit Description",value=str(selected_row['Description']))
        edited_amount = st.number_input("Edit Amount",min_value=0.0,value=float(selected_row['Amount']))

        categories = [
            "Food",
            "Travel",
            "Bills",
            "Entertainment",
            "Healthcare",
            "Transport",
            "Grocery",
            "Clothing",
            "Electronics",
            "Online Shopping"
        ]
        current_category = str(selected_row['Category'])

        if current_category not in categories:
            categories.append(current_category)
        edited_category = st.selectbox("Edit Category",categories,index=categories.index(current_category))

        # BUTTONS
        col1, col2 = st.columns(2)

        # UPDATE BUTTON
        with col1:
            if st.button("Update Expense"):
                expense_df.loc[row_index,'Date'] = edited_date
                expense_df.loc[row_index,'Description'] = edited_description
                expense_df.loc[row_index,'Amount'] = edited_amount
                expense_df.loc[row_index,'Category'] = edited_category

                # Save CSV
                try:
                    expense_df.to_csv("expenses.csv",index=False,mode='w')
                    st.success("Expense Updated Successfully")
                except Exception as e:
                    st.error(e)

        # DELETE BUTTON
        with col2:
            if st.button("Delete Expense"):
                expense_df = expense_df.drop(row_index)
                expense_df = expense_df.reset_index(drop=True)

                # Save CSV
                try:
                    expense_df.to_csv("expenses.csv",index=False,mode='w')
                    st.success("Expense Deleted Successfully")
                except Exception as e:
                    st.error(e)

    except Exception as e:
        st.error(e)
    
##################################################

#DASHBOARD
#Load Data
if menu == "📊 Dashboard":
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#111827,#1F2937);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
    ">
        <h1 style="color:white;">
            💰 AI Expense Tracker
        </h1>
        <p style="color:#D1D5DB;font-size:18px;">
            Smart financial insights powered by AI and Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.header("Expense Dashboard")

    try:
        df = pd.read_csv("expenses.csv")
    except:
        df = pd.DataFrame()

    # Convert Amount Column to Numeric
    df['Amount'] = pd.to_numeric(df['Amount'],errors='coerce')

    # Remove Missing Amount Rows
    df = df.dropna(subset=['Amount'])

    #Date processing
    df["Date"] = pd.to_datetime(df["Date"],format='mixed',dayfirst=True,errors='coerce')
    df["Month"] = df["Date"].dt.strftime('%Y-%m')
    
    # MONTH FILTER
    available_months = sorted(df['Month'].dropna().unique(),reverse=True)

    month_options = {
    month: pd.to_datetime(month).strftime('%B %Y')
    for month in available_months
    }

    selected_month = st.selectbox("Select Month",available_months,format_func=lambda x: month_options[x])
    selected_month_df = df[df['Month'] == selected_month]
    if selected_month_df.empty:
        st.warning("No expense data available for selected month")
        st.stop()

    #Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="card blue-card">
            <h3>💰 Total Expenses</h3>
            <h1>₹ {selected_month_df['Amount'].sum():.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card green-card">
            <h3>🔥 Highest Expense</h3>
            <h1>₹ {selected_month_df['Amount'].max():.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card orange-card">
            <h3>📦 Transactions</h3>
            <h1>{len(selected_month_df)}</h1>
        </div>
        """, unsafe_allow_html=True)

    #Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Charts","🧠 Insights","📥 Reports"])

    with tab1:
        #Pie Chart
        st.subheader("Category-wise Spending")
        category_spend = selected_month_df.groupby('Category')['Amount'].sum()
        if category_spend.empty:
            st.warning("No data available")
        elif len(category_spend) == 1:
            st.info("Pie chart not shown for single category filter")
        else:
            fig1 = px.pie(values=category_spend.values,names=category_spend.index,title="Category-wise Spending",hole=0.4)
            fig1.update_layout(template="plotly_dark",paper_bgcolor="#0E1117",plot_bgcolor="#0E1117")
            st.plotly_chart(fig1,use_container_width=True)

        #Bar Chart
        st.subheader("Expenses by Category")
        if len(category_spend) == 1:
            st.info("Only one category available for visualization")
            st.dataframe(category_spend)
        else:
            bar_df = category_spend.reset_index()
            fig2 = px.bar(bar_df,x='Category',y='Amount',title="Expenses by Category",text_auto=True)
            fig2.update_layout(template="plotly_dark",paper_bgcolor="#0E1117",plot_bgcolor="#0E1117")
            st.plotly_chart(fig2,use_container_width=True)

        #Monthly Trend
        st.subheader("Monthly Expense Trend")

        #Month column creation
        monthly_expense = df.groupby('Month')['Amount'].sum()

        if len(monthly_expense) <= 1:
            st.info("Not enough monthly data for trend visualization")
            st.dataframe(monthly_expense,use_container_width=True,hide_index=True)
        else:
            monthly_df = monthly_expense.reset_index()
            monthly_df.columns = ['Month', 'Amount']
            fig3 = px.line(monthly_df,x='Month',y='Amount',markers=True,title="Monthly Expense Trend")
            fig3.update_layout(template="plotly_dark",paper_bgcolor="#0E1117",plot_bgcolor="#0E1117")
            st.plotly_chart(fig3,use_container_width=True)

        #Monthly Trend Breakdown
        st.subheader("Monthly Category Breakdown")
        monthly_category = df.groupby(['Month', 'Category'])['Amount'].sum().unstack()
        st.dataframe(monthly_category.fillna(0),use_container_width=True,hide_index=True)

        monthly_category_reset = monthly_category.reset_index()
        fig4 = px.bar(monthly_category_reset,x='Month',y=monthly_category.columns,title="Monthly Category Breakdown",barmode='stack')
        fig4.update_layout(template="plotly_dark",paper_bgcolor="#0E1117",plot_bgcolor="#0E1117")
        st.plotly_chart(fig4,use_container_width=True)

    with tab2:
        st.subheader("AI Spending Insights")
        #Insight 1 - top Spending Category
        top_category = category_spend.idxmax()
        top_amount = category_spend.max()
        st.info(
            f"You spend the most on {top_category} "
            f"(₹ {top_amount:.2f})"
        )

        #Insight 2 - Average Expense
        average_expense = selected_month_df['Amount'].mean()
        st.info(
            f"Average transaction amount: "
            f"₹ {average_expense:.2f}"
        )

        #Insight 3 - High Spending Warning
        if average_expense > 1000:
            st.warning(
                "Your average spending is relatively high"
            )
            
        #insight 4 - Low expense Category
        lowest_category = category_spend.idxmin()
        lowest_amount = category_spend.min()
        st.info(
            f"Least spending category: "
            f"{lowest_category} "
            f"(₹ {lowest_amount:.2f})"
        )

        #############################################
        #BUDGET ALERTS
        st.subheader("Budget Alerts")
        budgets = {
            "Food": 5000,
            "Travel": 3000,
            "Bills": 8000,
            "Entertainment": 2000,
            "Healthcare": 4000
        }

        # Current Spending By Category
        category_totals = selected_month_df.groupby('Category')['Amount'].sum()

        # Compare With Budget
        for category, limit in budgets.items():
            spent = category_totals.get(category, 0)
            if spent > limit:
                st.error(
                    f"⚠ {category} budget exceeded! "
                    f"Spent ₹ {spent:.2f} "
                    f"(Limit ₹ {limit})"
                )
            else:
                remaining = limit - spent
                st.success(
                    f"✅ {category}: "
                    f"₹ {remaining:.2f} remaining"
                )

        ##################################################
        # ANOMALY DETECTION
        st.subheader("Overspending Alerts")

        # Calculate Threshold
        average_amount = selected_month_df['Amount'].mean()
        std_amount = selected_month_df['Amount'].std()
        threshold = average_amount + (2 * std_amount)

        # Detect Large Expenses
        anomalies = selected_month_df[selected_month_df['Amount'] > threshold]

        # Display Results
        if anomalies.empty:
            st.success("✅ No unusual spending detected")

        else:
            st.warning("⚠ Unusually high expenses detected")
            st.dataframe(
                anomalies[
                    [
                        'Date',
                        'Description',
                        'Amount',
                        'Category'
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        ###################################################3
        #MONTHLY EXPENSE PREDICTION
        st.subheader("Next Month Expense Prediction")

        # Prepare Monthly Data
        monthly_data = df.groupby('Month')['Amount'].sum().reset_index()
        monthly_data = monthly_data.sort_values('Month')

        # Need minimum data
        if len(monthly_data) < 2:
            st.info("Not enough monthly data for prediction")
        else:
            # Create Numeric Month Index
            monthly_data['Month_Index'] = range(1,len(monthly_data) + 1)
            X = monthly_data[['Month_Index']]
            y = monthly_data['Amount']

            # Train Model
            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # Predict Next Month
            next_month = [[len(monthly_data) + 1]]
            predicted_expense = model_lr.predict(next_month)[0]
            last_month = monthly_data['Month'].iloc[-1]
            st.success(
                f"Predicted expense after {last_month}: "
                f"₹ {predicted_expense:.2f}"
            )

        ###################################################
        # SMART AI RECOMMENDATIONS
        st.subheader("🤖 Smart AI Recommendations")

        # Previous Month Comparison
        all_months = sorted(df['Month'].dropna().unique())
        current_index = all_months.index(selected_month)
        if current_index > 0:
            previous_month = all_months[current_index - 1]
            previous_df = df[df['Month'] == previous_month]
            current_total = selected_month_df['Amount'].sum()
            previous_total = previous_df['Amount'].sum()
            difference = current_total - previous_total

            if previous_total > 0:
                if difference > 0:
                    percent = (difference / previous_total) * 100
                    st.warning(
                        f"⚠ Spending increased by "
                        f"{percent:.1f}% compared to {previous_month}"
                    )
                elif difference < 0:
                    percent = (abs(difference) / previous_total) * 100
                    st.success(
                        f"✅ Spending reduced by "
                        f"{percent:.1f}% compared to {previous_month}"
                    )
                else:
                    st.info("Spending unchanged from previous month")

        # Dominating Category Detection
        top_percentage = (top_amount / selected_month_df['Amount'].sum()) * 100
        if top_percentage > 50:
            st.error(
                f"⚠ {top_category} contributes "
                f"{top_percentage:.1f}% of your monthly expenses"
            )

        # Financial Health Score
        st.subheader("💡 Financial Health Score")
        score = 100
        if average_expense > 3000:
            score -= 20
        if len(anomalies) > 0:
            score -= 20
        if top_percentage > 50:
            score -= 15
        score = max(score, 0)
        if score >= 80:
            st.success(f"Excellent Financial Health: {score}/100")
        elif score >= 60:
            st.warning(f"Moderate Financial Health: {score}/100")
        else:
            st.error(f"Poor Financial Health: {score}/100")
          

    with tab3:
        #Download CSV Button
        csv = selected_month_df.to_csv(index=False)
        st.download_button("Download Expense Report(csv)",csv,"expenses.csv","text/csv")

        # PDF REPORT DOWNLOAD
        pdf_data = generate_pdf_report(selected_month_df,selected_month)
        st.download_button(label="📄 Download PDF Report",data=pdf_data,file_name=f"expense_report_{selected_month}.pdf",mime="application/pdf")
