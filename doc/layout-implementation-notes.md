## Table Of Contents
- [Flow Chart Legend](#Flow-Chart-Legend)
- [FlowBox Hierarchy and Composition](#FlowBox-Hierarchy-and-Composition)
- [GridBox Hierarchy & Composition](#GridBox-Hierarchy-and-Composition)
- [Sizable Nodes Top Level Hierarchy](#Sizable-Nodes-Top-Level-Hierarchy)
- [Separators Hierarchy](#Separators-Hierarchy)
- [AlignBox Hierarchy and Composition](#AlignBox-Hierarchy-and-Composition)
- [LayoutLine Hierarchy](#LayoutLine-Hierarchy)
- [ManualConstraint](#ManualConstraint)


## Flow Chart Legend:
- Inheritance: _______
- Mixin: _ _ _ _ _
- Composition: _ _ {{VARIABLE}} _ >
- Optional Composition: --{{VARIABLE}}*_>

## FlowBox Hierarchy and Composition
```mermaid
flowchart TD
    LayoutCell --- MarginLayoutCell
    MarginLayoutCell --- FlowCell
    FlowCell -.cells.-> FlowConstraint
    
    MarginLayoutConfigurable --- FlowConfigurable
    FlowConfigurable -.- FlowCell & FlowConstraint
    
    LayoutConstraint --- NodeLayoutConstraint
    NodeLayoutConstraint --- FlowConstraint
    FlowConstraint -.constraint.->FlowBox

    Node --- LayoutNode
    LayoutNode --- FlowBox
    FlowBox --- VBox & HBox

 ```

## GridBox Hierarchy & Composition
```mermaid
flowchart TD
    LayoutCell --- MarginLayoutCell
    MarginLayoutCell --- GridCell
    GridCell -.cells.-> GridConstraint
    
    MarginLayoutConfigurable --- GridConfigurable
    GridConfigurable -.- GridCell & GridConstraint
    
    LayoutConstraint --- NodeLayoutConstraint
    NodeLayoutConstraint --- GridConstraint
    GridConstraint -.constraint.-> GridBox
    
    Node --- LayoutNode
    LayoutNode --- GridBox

 ```

## Sizable Nodes Top Level Hierarchy
```mermaid
flowchart TD
    Node --- WidthSizable & HeightSizable --- Sizable
```

## Separators Hierarchy
```mermaid
flowchart TD
    HeightSizable -.- VSeparator
    Line --- Separator --- HSeparator & VSeparator
    WidthSizable -.- HSeparator
```

## AlignBox Hierarchy and Composition
```mermaid
flowchart TD
    Node --- WidthSizable & HeightSizable --- Sizeable --- AlignBox
    AlignGroup -.group*.-> AlignBox
    AlignBox -.alignBoxes*.->AlignGroup
```

## LayoutLine Hierarchy
```mermaid
flowchart TD
    LayoutLine --- FlowLine & GridLine

```

## ManualConstraint
```mermaid
flowchart TD
    LayoutConstraint --- ManualConstraint & RelaxedManualConstraint
    LayoutCell -.cells.-> ManualConstraint & RelaxedManualConstraint
```
