 ```mermaid
 classDiagram
    LayoutConstraint <|-- NodeLayoutConstraint
    NodeLayoutConstraint <|-- FlowConfigurable~NodeLayoutConstraint~ 
    NodeLayoutConstraint <|-- GridConfigurable~NodeLayoutConstraint~ 
    FlowConfigurable~NodeLayoutConstraint~ <|-- FlowConstraint
    GridConfigurable~NodeLayoutConstraint~ <|-- GridConstraint
    class LayoutConstraint{
        +Node ancestorNode
    }
    class NodeLayoutConstraint{
        
    }
    class FlowConstraint{
        +Array~FlowCell~ displayedCells
        
    }
    class GridConstraint{
        +Array~GridCell~ displayedCells
    }
    class FlowConfigurable~NodeLayoutConstraint~{
    }
    class GridConfigurable~NodeLayoutConstraint~{
    }
    GridConstraint <.. LayoutNode~GridConstraint~: Generic
    Sizable~Node~ <|-- LayoutNode~GridConstraint~
    LayoutNode~GridConstraint~ <|-- GridBox
    class Sizable~Node~{
    }
    NodeLayoutConstraint *-- LayoutNode~GridConstraint~: Composition
    class LayoutNode~GridConstraint~{
        #_constraint
    }
    class GridBox{
    -Map~Node, GridCell~ cellMap
    }
    
 ```