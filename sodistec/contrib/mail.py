import smtplib

class Mailer:
    """
    Sending email when violation accourd
    """
    def __init__(self) -> None:
        # Set email for mail author
        self.email_address = ""
        self.password = ""

        self.PORT = 465
        self.server = smtplib.SMTP_SSL("smtp.gmail.com", self.PORT)

    def send(self, to_mail: str):
        # Send an email and return true if success,
        # otherwise false

        # Login to the mail provider
        self.server.login(self.email_address, self.password)

        # Sending the email
        self.server.sendmail(self.email_address, to_mail, "Pelanggaran Protokol Kesehatan!")
        self.server.quit()

