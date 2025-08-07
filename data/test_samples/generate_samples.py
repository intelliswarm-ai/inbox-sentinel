#!/usr/bin/env python3
"""
Generate realistic email samples for testing
"""
import os
import random
from pathlib import Path

# Legitimate email templates
LEGITIMATE_EMAILS = [
    # Business/Work emails
    {
        "subject": "Weekly Team Meeting - {date}",
        "sender": "{name}@{company}.com",
        "content": "Hi team,\n\nOur weekly meeting is scheduled for {date} at {time}. We'll discuss project updates and next steps.\n\nAgenda:\n- Sprint review\n- Budget discussions\n- Client feedback\n\nBest regards,\n{name}"
    },
    {
        "subject": "Project Update - Q{quarter} Progress",
        "sender": "{name}@{company}.com",
        "content": "Hello {recipient},\n\nI wanted to share our progress on the {project} project. We're currently {percent}% complete and on track for the deadline.\n\nKey achievements:\n- Completed development phase\n- User testing in progress\n- Documentation updated\n\nLet me know if you have any questions.\n\nBest,\n{name}"
    },
    # Personal emails
    {
        "subject": "Dinner plans for {day}",
        "sender": "{name}@gmail.com",
        "content": "Hi {recipient},\n\nAre we still on for dinner this {day}? I was thinking we could try that new restaurant downtown.\n\nLet me know what time works for you.\n\nThanks,\n{name}"
    },
    {
        "subject": "Happy Birthday!",
        "sender": "{name}@{email_provider}",
        "content": "Dear {recipient},\n\nWishing you a very happy birthday! Hope you have a wonderful day celebrating.\n\nLooking forward to seeing you soon.\n\nWarm wishes,\n{name}"
    },
    # Service/Business communications
    {
        "subject": "Invoice #{invoice_num} - Payment Received",
        "sender": "billing@{company}.com",
        "content": "Dear {recipient},\n\nThank you for your payment of ${amount} for invoice #{invoice_num}. Your payment has been processed successfully.\n\nInvoice Details:\n- Amount: ${amount}\n- Date: {date}\n- Service: {service}\n\nThank you for your business.\n\nBest regards,\nAccounts Team"
    },
    {
        "subject": "Appointment Confirmation - {date}",
        "sender": "appointments@{company}.com",
        "content": "Dear {recipient},\n\nThis confirms your appointment on {date} at {time} with Dr. {doctor}.\n\nLocation: {address}\n\nPlease arrive 15 minutes early and bring a valid ID.\n\nBest regards,\n{clinic_name}"
    },
    # Newsletter/Updates
    {
        "subject": "{company} Newsletter - {month} Edition",
        "sender": "newsletter@{company}.com",
        "content": "Hello {recipient},\n\nWelcome to our monthly newsletter! This month we're featuring:\n\n- New product launches\n- Customer success stories\n- Upcoming events\n- Industry insights\n\nRead more at our website.\n\nBest regards,\nThe {company} Team"
    },
    # Educational/Academic
    {
        "subject": "Course Registration Confirmation",
        "sender": "registrar@{university}.edu",
        "content": "Dear {recipient},\n\nYour registration for {course} has been confirmed.\n\nCourse Details:\n- Instructor: Prof. {instructor}\n- Schedule: {schedule}\n- Room: {room}\n- Credits: {credits}\n\nBooks and materials list is attached.\n\nBest regards,\nRegistrar's Office"
    },
    {
        "subject": "Event Invitation - {event}",
        "sender": "events@{company}.com",
        "content": "Dear {recipient},\n\nYou're invited to our {event} on {date} at {time}.\n\nEvent highlights:\n- Keynote speakers\n- Networking opportunities\n- Light refreshments\n\nPlease RSVP by {rsvp_date}.\n\nLooking forward to seeing you there!\n\n{organizer}"
    }
]

# Spam email templates
SPAM_EMAILS = [
    # Lottery/Prize scams
    {
        "subject": "CONGRATULATIONS! You've Won ${amount}!",
        "sender": "winner@{domain}.com",
        "content": "DEAR LUCKY WINNER,\n\nCONGRATULATIONS! Your email has been selected in our international lottery!\n\nYou have won ${amount}! To claim your prize, send us:\n- Full name\n- Address\n- Phone number\n- Bank details\n\nACT NOW! This offer expires in 24 hours!\n\nMr. {name}\nLottery Commission"
    },
    {
        "subject": "URGENT: Claim Your Prize NOW!",
        "sender": "prizes@{domain}.net",
        "content": "Winner #{number},\n\nYour ticket number {ticket} has won our grand prize of ${amount}!\n\nTo receive your money, click here and provide:\n- Social Security Number\n- Bank Account Details\n- Processing fee of ${fee}\n\nHURRY! Only {hours} hours left!\n\nPrize Department"
    },
    # Phishing attempts
    {
        "subject": "URGENT: Your Account Will Be Suspended",
        "sender": "security@{bank}-alert.com",
        "content": "IMMEDIATE ACTION REQUIRED\n\nYour {bank} account shows suspicious activity. To prevent suspension:\n\n1. Click this link immediately\n2. Verify your login credentials\n3. Update your security information\n\nFailure to act within 24 hours will result in permanent account closure.\n\nSecurity Team\n{bank} (This is not affiliated with the real {bank})"
    },
    {
        "subject": "Update Your Payment Information",
        "sender": "billing@{service}-update.com",
        "content": "Dear Customer,\n\nYour payment method for {service} has failed. Update immediately to avoid service interruption.\n\nClick here to:\n- Update credit card\n- Verify billing address\n- Confirm payment\n\nYour account will be suspended in 48 hours if not updated.\n\nCustomer Service"
    },
    # Get-rich-quick schemes
    {
        "subject": "Make ${amount} Per Week Working From Home!",
        "sender": "opportunity@{domain}.biz",
        "content": "WORK FROM HOME OPPORTUNITY!\n\nEarn ${amount} per week with just 2 hours of work daily!\n\nNo experience needed! We provide:\n- Full training\n- All materials\n- Guaranteed income\n\nOnly ${signup_fee} to get started!\n\nACT FAST! Limited spots available!\n\nClick here to join thousands of successful members!\n\nSuccess Coach {name}"
    },
    {
        "subject": "Secret Investment Strategy Revealed",
        "sender": "wealth@{domain}.info",
        "content": "EXCLUSIVE INVITATION\n\nDiscover the secret that made me ${amount} in {days} days!\n\nThis proven system:\n- Requires no experience\n- Works on autopilot\n- Guaranteed returns\n\nSpecial price: Only ${price} (normally ${normal_price})\n\nDon't miss out! Limited time offer!\n\nMillionaire Mentor {name}"
    },
    # Fake urgent notifications
    {
        "subject": "VIRUS DETECTED - Immediate Action Required",
        "sender": "security@{antivirus}-alert.com",
        "content": "CRITICAL SECURITY ALERT\n\n{virus_count} viruses detected on your computer!\n\nYour personal information is at risk!\n\nDownload our removal tool immediately:\n1. Click this link\n2. Install our software\n3. Run full system scan\n\nDon't wait! Your files may be corrupted!\n\nSecurity Team\n{antivirus} Protection"
    },
    {
        "subject": "Package Delivery Failed - Action Needed",
        "sender": "delivery@{courier}-notify.com",
        "content": "DELIVERY NOTIFICATION\n\nWe attempted to deliver package #{tracking} but failed.\n\nTo reschedule:\n1. Click tracking link\n2. Verify your address\n3. Pay delivery fee ${fee}\n\nPackage will be returned in 2 days if not claimed.\n\nDelivery Services\n{courier} Express"
    },
    # Romance/relationship scams
    {
        "subject": "You Have a New Message from {name}",
        "sender": "messages@{dating_site}.com",
        "content": "Hi Sweetheart,\n\nI saw your profile and you caught my eye! I'm {name}, a {profession} from {country}.\n\nI'm looking for someone special like you for a serious relationship.\n\nI have a business proposal that could benefit us both. I need a trustworthy person to help me transfer ${amount} from my late father's estate.\n\nIf you help me, I'll share 30% with you and we can meet in person.\n\nContact me privately: {email}\n\nWith love,\n{name}"
    },
    # Health/medical scams
    {
        "subject": "Lose {weight}lbs in {days} Days - Guaranteed!",
        "sender": "health@{domain}.org",
        "content": "BREAKTHROUGH DISCOVERY!\n\nLose {weight} pounds in just {days} days without diet or exercise!\n\nOur revolutionary pill:\n- Burns fat while you sleep\n- No side effects\n- Clinically proven\n- 100% natural\n\nSpecial offer: Only ${price} (Save ${savings}!)\n\nFREE shipping if you order in the next {hours} hours!\n\nDr. {doctor_name}\nWeight Loss Institute"
    }
]

# Sample data for variables
NAMES = ["John", "Sarah", "Mike", "Lisa", "David", "Emma", "Chris", "Anna", "James", "Maria", 
         "Robert", "Jennifer", "William", "Amanda", "Michael", "Jessica", "Daniel", "Ashley",
         "Matthew", "Stephanie", "Andrew", "Nicole", "Joshua", "Elizabeth", "Ryan", "Megan"]

COMPANIES = ["TechCorp", "DataSys", "InnovateInc", "GlobalTech", "FutureSoft", "CloudWorks", 
            "NetSolutions", "InfoTech", "SystemsPro", "DigitalHub", "SmartSystems", "TechFlow"]

EMAIL_PROVIDERS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "protonmail.com"]

DOMAINS = ["winning-prize", "secure-bank", "fast-money", "easy-cash", "quick-profit",
          "mega-win", "super-deal", "best-offer", "hot-deals", "money-maker"]

BANKS = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One"]
SERVICES = ["Netflix", "Spotify", "Amazon Prime", "PayPal", "Apple"]
UNIVERSITIES = ["StateUniversity", "TechCollege", "CityUniversity"]

def generate_legitimate_email(template_idx, email_idx):
    """Generate a legitimate email from template"""
    template = LEGITIMATE_EMAILS[template_idx % len(LEGITIMATE_EMAILS)]
    
    # Random data for template
    data = {
        "name": random.choice(NAMES),
        "recipient": random.choice(NAMES),
        "company": random.choice(COMPANIES),
        "email_provider": random.choice(EMAIL_PROVIDERS),
        "date": f"January {random.randint(1, 31)}, 2025",
        "time": f"{random.randint(9, 17)}:{random.choice(['00', '30'])}",
        "quarter": random.randint(1, 4),
        "project": random.choice(["Alpha", "Beta", "Phoenix", "Titan", "Nova"]),
        "percent": random.randint(60, 95),
        "day": random.choice(["Friday", "Saturday", "Sunday"]),
        "invoice_num": f"INV{random.randint(1000, 9999)}",
        "amount": f"{random.randint(100, 5000):.2f}",
        "service": "Professional Services",
        "doctor": f"Dr. {random.choice(NAMES)}",
        "address": f"{random.randint(100, 999)} Main St, City, State",
        "clinic_name": "Health Center",
        "month": random.choice(["January", "February", "March", "April"]),
        "university": random.choice(UNIVERSITIES),
        "course": f"CS {random.randint(100, 499)}",
        "instructor": random.choice(NAMES),
        "schedule": "MWF 10:00-11:00",
        "room": f"Room {random.randint(100, 400)}",
        "credits": random.randint(3, 4),
        "event": random.choice(["Annual Conference", "Tech Summit", "Networking Event"]),
        "rsvp_date": f"January {random.randint(10, 25)}, 2025",
        "organizer": random.choice(NAMES)
    }
    
    subject = template["subject"].format(**data)
    sender = template["sender"].format(**data)
    content = template["content"].format(**data)
    
    return {
        "filename": f"legitimate_{email_idx:03d}.txt",
        "subject": subject,
        "sender": sender,
        "content": content,
        "label": "ham"
    }

def generate_spam_email(template_idx, email_idx):
    """Generate a spam email from template"""
    template = SPAM_EMAILS[template_idx % len(SPAM_EMAILS)]
    
    # Random data for template
    data = {
        "name": random.choice(NAMES),
        "domain": random.choice(DOMAINS),
        "amount": f"{random.randint(10000, 1000000):,}",
        "number": random.randint(1000, 9999),
        "ticket": f"TK{random.randint(100000, 999999)}",
        "fee": random.randint(50, 500),
        "hours": random.randint(12, 48),
        "bank": random.choice(BANKS),
        "service": random.choice(SERVICES),
        "signup_fee": random.randint(50, 200),
        "days": random.randint(7, 30),
        "price": random.randint(50, 300),
        "normal_price": random.randint(500, 1000),
        "virus_count": random.randint(15, 50),
        "antivirus": "SecureGuard",
        "tracking": f"TR{random.randint(100000000, 999999999)}",
        "courier": random.choice(["FedEx", "UPS", "DHL"]),
        "profession": random.choice(["doctor", "engineer", "businessman"]),
        "country": random.choice(["USA", "UK", "Canada", "Australia"]),
        "email": f"{random.choice(NAMES).lower()}@{random.choice(EMAIL_PROVIDERS)}",
        "dating_site": "LoveConnect",
        "weight": random.randint(20, 50),
        "savings": random.randint(100, 500),
        "doctor_name": f"Dr. {random.choice(NAMES)}"
    }
    
    subject = template["subject"].format(**data)
    sender = template["sender"].format(**data)
    content = template["content"].format(**data)
    
    return {
        "filename": f"spam_{email_idx:03d}.txt",
        "subject": subject,
        "sender": sender,
        "content": content,
        "label": "spam"
    }

def create_forwarded_email(email_data):
    """Convert email data to forwarded Gmail format"""
    return f"""---------- Forwarded message ---------
From: {email_data['sender']}
Date: Mon, {random.randint(1, 28)} Jan 2025 at {random.randint(9, 17)}:{random.choice(['00', '15', '30', '45'])}
Subject: {email_data['subject']}
To: testuser@example.com

{email_data['content']}
"""

def main():
    """Generate all email samples"""
    base_path = Path("./legitimate")
    spam_path = Path("./spam")
    
    print("Generating 200 legitimate emails...")
    for i in range(200):
        template_idx = i % len(LEGITIMATE_EMAILS)
        email = generate_legitimate_email(template_idx, i)
        
        email_content = create_forwarded_email(email)
        
        with open(base_path / email['filename'], 'w', encoding='utf-8') as f:
            f.write(email_content)
    
    print("Generating 800 spam emails...")
    for i in range(800):
        template_idx = i % len(SPAM_EMAILS)
        email = generate_spam_email(template_idx, i)
        
        email_content = create_forwarded_email(email)
        
        with open(spam_path / email['filename'], 'w', encoding='utf-8') as f:
            f.write(email_content)
    
    print("Sample generation complete!")
    print(f"Created {len(os.listdir(base_path))} legitimate emails")
    print(f"Created {len(os.listdir(spam_path))} spam emails")

if __name__ == "__main__":
    main()