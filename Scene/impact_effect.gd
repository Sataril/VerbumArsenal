extends AnimatedSprite2D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	play() 
	animation_finished.connect(on_animation_finished)

func on_animation_finished() -> void:
	queue_free()
