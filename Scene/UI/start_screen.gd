extends Control

@onready var start_button: Button = $StartButton
@export var game_scene: PackedScene


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	start_button.pressed.connect(_on_start_button_pressed)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _on_start_button_pressed (button_index: int = -1) -> void:
	# 切换到游戏主场景（假设你的主场景保存为 MainGame.tscn）
	get_tree().change_scene_to_file("res://Scene/main.tscn")
