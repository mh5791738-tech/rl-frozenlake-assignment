import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(OUTPUT_DIR, 'technical_report.pdf')
PLOT_PATH = os.path.join(OUTPUT_DIR, 'training_progress.png')
GIF_PATH = os.path.join(OUTPUT_DIR, 'trained_agent.gif')

styles = getSampleStyleSheet()


def build_report():
    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )
    story = []

    story.append(Paragraph('Custom Reinforcement Learning Agent Implementation using Farama Gymnasium', styles['Title']))
    story.append(Spacer(1, 10))
    story.append(Paragraph('Course: Topics in AI | Assignment No. I/CLO-I', styles['Heading2']))
    story.append(Spacer(1, 18))

    story.append(Paragraph('<b>1. Problem Description</b>', styles['Heading2']))
    story.append(Paragraph(
        'The selected environment is FrozenLake-v1 from Gymnasium with the slippery setting disabled. The agent observes a discrete state index representing the current tile in a 4x4 grid. The action space contains four possible moves: left, down, right, and up. The reward structure is sparse: the agent receives 1.0 upon reaching the goal tile and 0.0 elsewhere, with falling into a hole causing termination. This environment is a classic small-scale control problem that requires learning a policy that avoids unsafe tiles while navigating to the reward state.'
    , styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph('<b>2. Methodology</b>', styles['Heading2']))
    story.append(Paragraph(
        'The implemented method is tabular Q-learning, a model-free RL algorithm that estimates the action-value function Q(s, a). The update rule used is: Q(s, a) ← Q(s, a) + α[r + γ max_a&#39; Q(s&#39;, a&#39;) − Q(s, a)], where α is the learning rate, γ is the discount factor, and r is the immediate reward. The agent follows an ε-greedy policy during training, balancing exploration and exploitation. Once the Q-table converges, the greedy policy arg max_a Q(s, a) is used for evaluation and policy execution.'
    , styles['BodyText']))
    story.append(Spacer(1, 12))

    hyperparams = Table([
        ['Parameter', 'Value'],
        ['States', '16'],
        ['Actions', '4'],
        ['Learning Rate (α)', '0.8'],
        ['Discount Factor (γ)', '0.99'],
        ['Initial ε', '1.0'],
        ['Minimum ε', '0.05'],
        ['ε Decay', '0.995'],
        ['Episodes', '3000'],
        ['Max Steps per Episode', '200'],
    ], colWidths=[220, 140])
    hyperparams.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2F80ED')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(Paragraph('<b>Experimental Setup & Hyperparameters</b>', styles['Heading2']))
    story.append(Spacer(1, 8))
    story.append(hyperparams)
    story.append(Spacer(1, 18))

    story.append(Paragraph('<b>3. Results & Discussion</b>', styles['Heading2']))
    story.append(Paragraph(
        'The training curve shows rapid learning in the early episodes, followed by stable convergence near the optimal policy. The agent reaches a perfect evaluation score over 200 test episodes, yielding an average return of 1.0 and a success rate of 100%. The learned behavior avoids holes and consistently reaches the goal tile. The main observed challenge was exploration in the early phase, which is why the agent starts with a relatively high ε and decays it gradually. This encourages sufficient exploration before exploitation takes over.'
    , styles['BodyText']))
    story.append(Spacer(1, 10))
    try:
        img = Image(PLOT_PATH, width=480, height=240)
        story.append(img)
    except Exception:
        story.append(Paragraph('Training plot image unavailable.', styles['BodyText']))
    story.append(Spacer(1, 16))

    story.append(Paragraph('<b>4. Conclusion</b>', styles['Heading2']))
    story.append(Paragraph(
        'The Q-learning implementation successfully solved the FrozenLake environment. The trained policy demonstrates stable convergence, robust performance, and a clear interpretation of the learned state-action values. This assignment confirms the effectiveness of tabular RL methods for small discrete environments and provides a reproducible baseline for further experimentation with deeper algorithms or more complex environments.'
    , styles['BodyText']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f'Asset: {os.path.basename(GIF_PATH)}', styles['BodyText']))

    doc.build(story)
    print(f'Report saved to: {REPORT_PATH}')


if __name__ == '__main__':
    build_report()
