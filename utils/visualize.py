import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def visualize_bev(df, sample_frames=500, lane_width=12, view_length=500):
    fig, ax = plt.subplots(figsize=(18,6))
    
    class_colors = {1: '#3498db', 2: '#2ecc71', 3: '#e74c3c'}
    frames = sorted(df["Frame_ID"].unique())[:sample_frames]

    def update(frame):
        ax.cla() # Clear everything to redraw the frame
        
        # Set Fixed Viewport
        ax.set_xlim(0, view_length)
        ax.set_ylim(0, df["Lane_ID"].max() * lane_width + 5)
        ax.set_aspect('equal')
        
        # Re-draw Lanes (Static Positions)
        for y in range(0, df["Lane_ID"].max() * lane_width, lane_width):
            ax.axhline(y=y, color='lightgray', linestyle='--', zorder=0)

        frame_df = df[df["Frame_ID"] == frame]

        #  Draw Vehicles
        for _, veh in frame_df.iterrows():
            # Only draw if within our fixed camera view to save processing
            if 0 <= veh['Local_Y'] <= view_length:
                rect = patches.Rectangle(
                    (veh['Local_Y'] - veh['v_len']/2, veh['Local_X'] - veh['v_width']/2), 
                    veh['v_len'], 
                    veh['v_width'],
                    facecolor=class_colors.get(veh['v_class'], 'gray'),
                    edgecolor='black',
                    alpha=0.7,
                    zorder=3
                )
                ax.add_patch(rect)

                # Add Vehicle ID Text
                # place it at (Local_Y, Local_X) which is the center
                ax.text(
                    veh['Local_Y'], 
                    veh['Local_X'], 
                    str(int(veh['Veh_ID'])),
                    color='black',
                    fontsize=8,
                    fontweight='bold',
                    ha='center', 
                    va='center',
                    zorder=4,
                    #Add a subtle white glow for readability
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5)
                )
        
        ax.set_xlabel("Longitudinal Position (ft)")
        ax.set_ylabel("Lateral Position (ft)")
        ax.set_title(f"NGSIM Frame: {frame} | Green: Auto, Blue: Moto, Red: Truck")

    # blit=False is necessary when using ax.cla()
    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False) # change interval based on sample_frames to avoid lagging
    plt.close()
    return HTML(anim.to_jshtml())
