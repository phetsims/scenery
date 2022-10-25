 ```mermaid
 classDiagram
    LayoutConstraint <|-- NodeLayoutConstraint
    NodeLayoutConstraint <|-- FlowConfigurable~NodeLayoutConstraint~ 
    NodeLayoutConstraint <|-- GridConfigurable~NodeLayoutConstraint~ 
    FlowConfigurable~NodeLayoutConstraint~ <|-- FlowConstraint
    GridConfigurable~NodeLayoutConstraint~ <|-- GridConstraint
    class LayoutConstraint{
        +Node ancestorNode
        -_layoutCount
        -_layoutAttemptDuringLock
        -_enabled
        -_updateLayoutListener
        +addNode(node, addLock = true)
        +removeNode(node)
        -layout()
        +lock()
        +unlock()
        +validateLocalPreferredWidth(layoutContainer)
        +validateLocalPreferredHeight(layoutContainer)
        +updateLayout()
        +createLayoutProxy(node)
    }
    class NodeLayoutConstraint{
        +layoutBoundsProperty
        +preferredWidthProperty
        +preferredHeightProperty
        +minimumWidthProperty
        +minimumHeightProperty
        +layoutOriginProperty
        #filterLayoutCells()
        +setProxyPreferredSize()
        +setProxyMinSide()
        +setProxyOrigin()
    }
    class FlowConstraint{
        -Array~FlowCell~ cells
        -_justify
        -_justifyLines
        -_wrap
        -_spacing
        -_lineSpacing
        +Array~FlowCell~ displayedCells
        #layout()
        +justify()
        +justifyLines()
        +wrap()
        +spacing()
        +lineSpacing()
        +insertCell()
        +removeCell()
        +reorderCells()
        +getPreferredProperty
    }
    class GridConstraint{
        -Set~GridCell~ cells
        +Array~GridCell~ displayedCells
        +displayedLines
        -_spacing
        #layout()
        +spacing()
        +xSpacing()
        +ySpacing()
        +addCell()
        +removeCell()
        +getIndices()
        +getCell()
        +getCellFromNode()
        +getCells()
    }
    class FlowConfigurable~NodeLayoutConstraint~{
        #_orientation
        +_align
        +_stretch
        +_grow
        +setConfigToBaseDefault()
        +setConfigtoInherit()
    }
    class GridConfigurable~NodeLayoutConstraint~{
        #_orientation
        +_align
        +_stretch
        +_grow
        +setConfigToBaseDefault()
        +setConfigtoInherit()
    }
    GridConstraint <.. LayoutNode~GridConstraint~
    Sizable~Node~ <|-- LayoutNode~GridConstraint~
    LayoutNode~GridConstraint~ <|-- GridBox
    class Sizable~Node~{
        +preferredSize()
        +localPreferredSize()
        +minimumSize()
        +localMinimumSize()
        +sizable()
        +mixesSizable()
        +validateLocalPreferredSize()
        +mutate()
        +calculateLocalPreferredWidth()
        +calculateLocalPreferredHeight()
    }
    class LayoutNode~GridConstraint~{
        #_constraint
        +layoutOriginProperty
        #linkLayoutBounds()
        +setExcludeInvisibleChildrenFromBounds(excludeInvisibleChildrenFromBounds)
        +setChildren(children)
        +updateLayout()
        +resize()
        +layoutOrigin()
        +constraint()
    }
    class GridBox{
        -_cellMap
        -_autoRows
        -_autoColumns
        -_autoLockCount
        -onChildInserted
        -onChildRemoved
        -onChildVisibilityToggled
        +setLines()
        +getLines()
        +rows()
        +columns()
        +getNodeAt(row, column)
        +getRowOfNode(node)
        +getColumnOfNode(node)
        +getNodesInRow(index)
        +getNodesInColumn(index)
        +addRow()
        +addColumn()
        +insertRow()
        +insertColumn()
        +removeRow()
        +removeColumn()
        +autoRows()
        +autoColumns()
        -updateAutoLines(orientation, value)
        +setChildren(children)
        +spacing()
        +xSpacing()
        +ySpacing()
        +xAlign()
        +yAlign()
        +grow()
        +stretch()
        +margin()
        +minContentWidth()
        +maxContentWidth()
        +minContentHeight()
        +maxContentHeight()
        +getHelperNode()
    }
    
 ```