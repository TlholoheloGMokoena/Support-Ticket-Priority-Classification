"""Creates a SYNTHETIC support-ticket dataset (data/Support_tickets.csv) so the project runs without private data.
Replace the file with your real export (same column names) and re-run train_model.py."""
import random
import pandas as pd
from support_ticket_classifier import DATA_PATH

random.seed(42)
PHRASES = {  # category -> priority -> issue descriptions
 "Billing": {
  "High": ["payments are failing for all of our customers at checkout", "we were charged several times and refunds have not arrived for many customers", "the payment gateway is down and no transactions go through"],
  "Medium": ["my invoice shows an incorrect amount this month", "a bank transfer is delayed and still not reflected on my account", "my subscription renewal charged me for the wrong plan"],
  "Low": ["could you explain how the billing cycle works", "how can I download last year's invoices", "do you offer a discount for annual payment"]},
 "Technical Support": {
  "High": ["the whole platform is down and nobody on our team can log in", "we lost data after the last sync and files are missing", "the application crashes on start for every user with a critical error"],
  "Medium": ["the export feature is not working correctly for me", "I keep getting an error message when uploading reports", "the mobile app is slow and freezes from time to time"],
  "Low": ["is there a keyboard shortcut for search", "I suggest adding a dark mode to the interface", "how do I enable notifications for new messages"]},
 "Account Issues": {
  "High": ["my account was hacked and someone changed the password", "there are unauthorised logins from unknown locations on our account", "a security breach exposed customer details stored in our account"],
  "Medium": ["I cannot reset my password because the email never arrives", "my two-factor code is rejected every time", "my account is locked after several login attempts"],
  "Low": ["how do I change the email address on my profile", "can I rename my username", "where can I update my company name"]},
 "Product Enquiries": {
  "High": ["we need to know whether the product is affected by the safety recall", "the product overheated and we have a safety concern"],
  "Medium": ["the feature described on your website does not work as documented", "the product I received is missing a component"],
  "Low": ["does the premium plan include API access", "what is the difference between the basic and pro versions", "when will the new version be released"]},
 "General Queries": {
  "High": ["we received a legal notice that needs a reply from your company today"],
  "Medium": ["nobody has answered my earlier request for a week", "I was promised a callback but have not heard back"],
  "Low": ["what are your opening hours", "I would like to give some feedback on your service", "where can I find your privacy policy"]},
}
TYPES = {"Billing": ["Request", "Problem"], "Technical Support": ["Incident", "Problem"], "Account Issues": ["Incident", "Request"],
         "Product Enquiries": ["Request", "Change"], "General Queries": ["Request"]}
GREET = ["Hello team,", "Hi,", "Good day,", "Dear support,"]
EXTRA = ["I tried to fix it myself without success.", "This started earlier today.", "Please let me know what you need from me.", "", "I have attached nothing yet."]
CLOSE = ["Thanks in advance.", "Regards.", "Looking forward to your reply.", "Thank you."]
URGENT = ["urgent", "asap", "immediately", "please help"]

rows = []
for i in range(3000):
    cat = random.choices(list(PHRASES), weights=[25, 30, 20, 15, 10])[0]
    prio = random.choices(["High", "Medium", "Low"], weights=[15, 35, 50])[0]
    phrase = random.choice(PHRASES[cat][prio])
    p_urgent = {"High": 0.5, "Medium": 0.15, "Low": 0.05}[prio]
    urgent = (random.choice(URGENT) + ". ") if random.random() < p_urgent else ""
    body = f"{random.choice(GREET)} {urgent}{phrase[0].upper() + phrase[1:]}. {random.choice(EXTRA)} {random.choice(CLOSE)}"
    if random.random() < 0.10:  # 10% label noise so evaluation is not unrealistically perfect
        prio = random.choice([p for p in ["High", "Medium", "Low"] if p != prio])
    rows.append({"subject": " ".join(phrase.split()[:6]).capitalize(), "body": body, "answer": "Thank you for contacting us, we are looking into it.",
                 "type": random.choice(TYPES[cat]), "queue": cat, "priority": prio, "language": "en", "tags": cat.lower().replace(" ", "-"),
                 "created_at": (pd.Timestamp("2026-07-01") + pd.Timedelta(days=random.randint(0, 120))).date().isoformat(),
                 "data_source": "synthetic-sample"})
DATA_PATH.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(DATA_PATH, index=False)
print(f"Wrote {len(rows)} synthetic tickets to {DATA_PATH}")
