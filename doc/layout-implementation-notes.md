 ```mermaid
 classDiagram
    class MarginLayoutConfigurable
    class MarginLayoutCell
    
    LayoutCell<|--MarginLayoutCell
    
    MarginLayoutCell<|--MarginLayoutConfigurable
    NodeLayoutConstraint<|--MarginLayoutConfigurable
    
    MarginLayoutConfigurable<|--GridConfigurable
    MarginLayoutConfigurable<|--FlowConfigurable
    
    class GridConfigurable
    class FlowConfigurable
    
    GridConfigurable <|-- GridConstraint:( NodeLayoutConstraint ) 
    FlowConfigurable <|-- FlowConstraint: ( NodeLayoutConstraint )
    
    LayoutConstraint <|-- NodeLayoutConstraint
    NodeLayoutConstraint <.. GridConfigurable:Generic 
    NodeLayoutConstraint <.. FlowConfigurable:Generic 

    class LayoutConstraint{
        +Node ancestorNode
    }
    class NodeLayoutConstraint
    class GridConstraint{
        +Array~GridCell~ displayedCells
    }
    class FlowConstraint{
        +Array~FlowCell~ displayedCells    
    }

    GridConstraint <.. LayoutNode: Generic
    FlowConstraint <.. LayoutNode: Generic
    
    class Sizable
    Node<|--HeightSizable
    HeightSizable<|--WidthSizable: ( Node )
    WidthSizable<|--Sizable: ( HeighSizable )
    Sizable <|-- LayoutNode: ( Node )
    
    class LayoutNode{
    <<Abstract>>
    }
    LayoutNode <|-- GridBox: <>GridConstraint
    LayoutNode <|-- FlowBox: <>FlowConstraint
    
    class GridBox
    class FlowBox
    
    GridConfigurable<|--GridCell: ( MarginLayoutCell )
    MarginLayoutCell<..GridConfigurable: Generic
    
    FlowConfigurable<|--FlowCell: ( MarginLayoutCell )
    MarginLayoutCell<..FlowConfigurable: Generic
    
    class GridCell
    class FlowCell
    
    GridCell--*GridConstraint:Composition
    FlowCell--*FlowConstraint:Composition
    
    class VBox
    class HBox
    FlowBox <|-- VBox
    FlowBox <|-- HBox
    
    class AlignBox
    Sizable <-- AlignBox: ( Node )
 ```