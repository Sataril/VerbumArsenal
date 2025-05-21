extends Node2D

@export var enemy_scene: PackedScene # 敌人生成场景
@export var spawn_radius: float = 64 # 刷新点半径
@export var minimum_spawn_delay = 2  # 最小间隔
@export var maximum_spawn_delay =5   # 最大间隔

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


func on_time_out() -> void:
	pass
	
func get_random_spwan_position() -> Vector2: 
	var theta = randf_range(0,2 * PI)
	var r = sqrt(randf_range(0, spawn_radius * spawn_radius))
	var x = global_position.x + r * cos(theta)
	var y = global_position.y + r * sin(theta)
	return Vector2(x,y)
