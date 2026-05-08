import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv()

message = Mail(
    from_email=os.environ["SENDGRID_FROM_EMAIL"],
    to_emails=os.environ["SENDGRID_FROM_EMAIL"],
    subject="LaunchMind test email",
    html_content="<p>Email agent is working!</p>"
)

sg = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
response = sg.send(message)
print(response.status_code)