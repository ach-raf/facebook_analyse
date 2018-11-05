import os
import sys
import json
from sys import argv
from bs4 import BeautifulSoup
from datetime import date
from datetime import datetime as dt
from pickle import dump

my_subject = sys.argv[1]
if not os.path.exists('./result/{0}/{0}_input'.format(my_subject)):
    os.makedirs('./result/{0}/{0}_input'.format(my_subject))
    os.makedirs('./result/{0}/{0}_plots'.format(my_subject))


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (dt, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def convert_date(x):
    """
    Converts Facebook date strings to datetime objects
    """
    return dt.strptime(x.split(' UTC')[0], '%A, %B %d, %Y at %I:%M%p')


def write_to_disk(subject, master):
    # Pickle the messages
    print('Saving pickled list of messages at result/{0}/{0}_input/{0}_messages.pkl'.format(subject))
    with open('./result/{0}/{0}_input/{0}_messages.pkl'.format(subject), 'wb') as outfile:
        dump(master, outfile)
    # reverse the list to have the first messages first
    reversed_master = list(reversed(master))
    # Json file
    print('Saving messages in JSON format at result/{0}/{0}_input/{0}_messages.json'.format(subject))
    with open('./result/{0}/{0}_input/{0}_messages.json'.format(subject), 'w') as outfile:
        json.dump(reversed_master, outfile, default=json_serial)
    # txt format
    print('Saving messages in txt format at result/{0}/{0}_input/{0}_messages.txt'.format(subject))
    file_to_write = open('./result/{0}/{0}_input/{0}_messages.txt'.format(subject), 'w', encoding='utf8')
    for entry in reversed_master:
        file_to_write.write(
            entry['time'].strftime('%#m/%#d/%y, %H:%M') + ' - ' + entry['sender'] + ': ' + entry['text'] + '\n')


def thread_parse(subject):
    """
    Parses message data from a HTML file containing one thread.
    """

    # Open and import file
    html = open('./chat/{0}.html'.format(subject), 'r', encoding='utf8')
    # creating a new file to keep the integrity of the original while doing the cleaning on the subject_clean.txt file
    re_write = open('./result/{0}/{0}_input/{0}_clean.html'.format(subject), 'w', encoding='utf8')
    # I realised there is a problem with they way fb format their html files
    # they have added a <p> tag before the body of the message so just remove it to access the message directly
    for line in html:
        temp = line.replace('<p><p>', '<p>').replace('</p></p>', '</p>')
        re_write.write(temp)
    # opening the clean chat
    html = open('./result/{0}/{0}_input/{0}_clean.html'.format(subject), 'r', encoding='utf8')

    # Create a BS object
    print('Parsing HTML (may take a while)...')
    soup = BeautifulSoup(html, 'lxml')

    # Collect the thread object from the HTML
    thread = soup.findAll('div', {'class': 'thread'})[0]

    # Collect all the paragraph text, these are the message bodies
    # and if it's an image get the src attribute
    texts = [message.img['src'] if message.img else message.text for message in thread.findAll('p')]

    # print the number of messages found on the chat
    print('Found {0} messages in this thread'.format(len(texts)))

    # Collect the message objects from the thread
    messages = thread.findAll('div', {'class': 'message'})

    # Get the message headers
    headers = [m.findNext('div', {'class': 'message_header'}) for m in messages]

    # Get the sender from the message headers
    users = [h.findNext('span', {'class': 'user'}).text for h in headers]

    # in the case of an empty user (usually due to blocked user or the user deactivated his/her account)
    # we change empty string to Facebook User
    users = ['Facebook User' if not user else user for user in users]

    # Get the times from the message headers and convert to datetime
    times = [convert_date(h.findNext('span', {'class': 'meta'}).text) for h in headers]

    # Create a list of dicts for all messages
    master = [{'sender': users[i],
               'time': times[i],
               'text': texts[i]}
              for i in range(len(texts))]

    write_to_disk(subject, master)
    return master


if __name__ == '__main__':
    # If ran independently, takes the HTML file path as input
    thread_parse(argv[1])
