// Copyright 2021-2023, University of Colorado Boulder

/**
 * A LayoutCell that has margins, and can be positioned and sized relative to those. Used for Flow/Grid layouts
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../../dot/js/Bounds2.js';
import Matrix3 from '../../../../dot/js/Matrix3.js';
import Utils from '../../../../dot/js/Utils.js';
import { Shape } from '../../../../kite/js/imports.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../../phet-core/js/OrientationPair.js';
import Tandem from '../../../../tandem/js/Tandem.js';
import { Font, TColor, LayoutAlign, LayoutCell, LayoutProxy, Node, NodeLayoutConstraint, NodePattern, Path, PressListener, Rectangle, RichText, scenery, Text } from '../../imports.js';

// Interface expected to be overridden by subtypes (GridCell, FlowCell)
export type MarginLayout = {
  _leftMargin: number | null;
  _rightMargin: number | null;
  _topMargin: number | null;
  _bottomMargin: number | null;
  _minContentWidth: number | null;
  _minContentHeight: number | null;
  _maxContentWidth: number | null;
  _maxContentHeight: number | null;
};

export type MarginLayoutConstraint = NodeLayoutConstraint & MarginLayout;

// NOTE: This would be an abstract class, but that is incompatible with how mixin constraints work in TypeScript
export default class MarginLayoutCell extends LayoutCell {

  private readonly _marginConstraint: MarginLayoutConstraint;

  private readonly preferredSizeSet: OrientationPair<boolean> = new OrientationPair<boolean>( false, false );

  // These will get overridden, they're needed since mixins have many limitations and we'd have to have a ton of casts
  // without these existing.
  // (scenery-internal)
  public _leftMargin!: number | null;
  public _rightMargin!: number | null;
  public _topMargin!: number | null;
  public _bottomMargin!: number | null;
  public _minContentWidth!: number | null;
  public _minContentHeight!: number | null;
  public _maxContentWidth!: number | null;
  public _maxContentHeight!: number | null;

  // (scenery-internal) Set to be the bounds available for the cell
  public lastAvailableBounds: Bounds2 = Bounds2.NOTHING.copy();

  // (scenery-internal) Set to be the bounds used by the cell
  public lastUsedBounds: Bounds2 = Bounds2.NOTHING.copy();

  /**
   * NOTE: Consider this scenery-internal AND protected. It's effectively a protected constructor for an abstract type,
   * but cannot be due to how mixins constrain things (TypeScript doesn't work with private/protected things like this)
   *
   * (scenery-internal)
   */
  public constructor( constraint: MarginLayoutConstraint, node: Node, proxy: LayoutProxy | null ) {
    super( constraint, node, proxy );

    this._marginConstraint = constraint;
  }

  /**
   * Positions and sizes the cell (used for grid and flow layouts)
   * (scenery-internal)
   *
   * Returns the cell's bounds
   */
  public reposition( orientation: Orientation, lineSize: number, linePosition: number, stretch: boolean, originOffset: number, align: LayoutAlign ): Bounds2 {
    // Mimicking https://www.w3.org/TR/css-flexbox-1/#align-items-property for baseline (for our origin)
    // Origin will sync all origin-based items (so their origin matches), and then position ALL of that as if it was
    // align:left or align:top (depending on the orientation).

    const preferredSize = ( stretch && this.isSizable( orientation ) ) ? lineSize : this.getMinimumSize( orientation );

    this.attemptPreferredSize( orientation, preferredSize );

    if ( align === LayoutAlign.ORIGIN ) {
      this.positionOrigin( orientation, linePosition + originOffset );
    }
    else {
      this.positionStart( orientation, linePosition + ( lineSize - this.getCellBounds()[ orientation.size ] ) * align.padRatio );
    }

    const cellBounds = this.getCellBounds();

    assert && assert( cellBounds.isFinite() );

    this.lastAvailableBounds[ orientation.minCoordinate ] = linePosition;
    this.lastAvailableBounds[ orientation.maxCoordinate ] = linePosition + lineSize;
    this.lastUsedBounds.set( cellBounds );

    return cellBounds;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveLeftMargin(): number {
    return this._leftMargin !== null ? this._leftMargin : this._marginConstraint._leftMargin!;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveRightMargin(): number {
    return this._rightMargin !== null ? this._rightMargin : this._marginConstraint._rightMargin!;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveTopMargin(): number {
    return this._topMargin !== null ? this._topMargin : this._marginConstraint._topMargin!;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveBottomMargin(): number {
    return this._bottomMargin !== null ? this._bottomMargin : this._marginConstraint._bottomMargin!;
  }

  /**
   * (scenery-internal)
   */
  public getEffectiveMinMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveLeftMargin : this.effectiveTopMargin;
  }

  /**
   * (scenery-internal)
   */
  public getEffectiveMaxMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveRightMargin : this.effectiveBottomMargin;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveMinContentWidth(): number | null {
    return this._minContentWidth !== null ? this._minContentWidth : this._marginConstraint._minContentWidth;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveMinContentHeight(): number | null {
    return this._minContentHeight !== null ? this._minContentHeight : this._marginConstraint._minContentHeight;
  }

  /**
   * (scenery-internal)
   */
  public getEffectiveMinContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMinContentWidth : this.effectiveMinContentHeight;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveMaxContentWidth(): number | null {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._marginConstraint._maxContentWidth;
  }

  /**
   * Returns the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveMaxContentHeight(): number | null {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._marginConstraint._maxContentHeight;
  }

  /**
   * (scenery-internal)
   */
  public getEffectiveMaxContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMaxContentWidth : this.effectiveMaxContentHeight;
  }

  /**
   * Returns the effective minimum size this cell can take (including the margins)
   * (scenery-internal)
   */
  public getMinimumSize( orientation: Orientation ): number {
    return this.getEffectiveMinMargin( orientation ) +
           Math.max( this.proxy.getMinimum( orientation ), this.getEffectiveMinContent( orientation ) || 0 ) +
           this.getEffectiveMaxMargin( orientation );
  }

  /**
   * Returns the effective maximum size this cell can take (including the margins)
   * (scenery-internal)
   */
  public getMaximumSize( orientation: Orientation ): number {
    return this.getEffectiveMinMargin( orientation ) +
           ( this.getEffectiveMaxContent( orientation ) || Number.POSITIVE_INFINITY ) +
           this.getEffectiveMaxMargin( orientation );
  }

  /**
   * Sets a preferred size on the content, obeying many constraints.
   * (scenery-internal)
   */
  public attemptPreferredSize( orientation: Orientation, value: number ): void {
    if ( this.proxy[ orientation.sizable ] ) {
      const minimumSize = this.getMinimumSize( orientation );
      const maximumSize = this.getMaximumSize( orientation );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      let preferredSize = value - this.getEffectiveMinMargin( orientation ) - this.getEffectiveMaxMargin( orientation );
      const maxSize = this.proxy.getMax( orientation );
      if ( maxSize !== null ) {
        preferredSize = Math.min( maxSize, preferredSize );
      }

      this._marginConstraint.setProxyPreferredSize( orientation, this.proxy, preferredSize );

      // Record that we set
      this.preferredSizeSet.set( orientation, true );
    }
  }

  /**
   * Unsets the preferred size (if WE set it)
   * (scenery-internal)
   */
  public unsetPreferredSize( orientation: Orientation ): void {
    if ( this.proxy[ orientation.sizable ] ) {
      this._marginConstraint.setProxyPreferredSize( orientation, this.proxy, null );
    }
  }

  /**
   * Sets the left/top position of the (content+margin) for the cell in the constraint's ancestor coordinate frame.
   * (scenery-internal)
   */
  public positionStart( orientation: Orientation, value: number ): void {
    const start = this.getEffectiveMinMargin( orientation ) + value;

    this._marginConstraint.setProxyMinSide( orientation, this.proxy, start );
  }

  /**
   * Sets the x/y value of the content for the cell in the constraint's ancestor coordinate frame.
   * (scenery-internal)
   */
  public positionOrigin( orientation: Orientation, value: number ): void {
    this._marginConstraint.setProxyOrigin( orientation, this.proxy, value );
  }

  /**
   * Returns the bounding box of the cell if it was repositioned to have its origin shifted to the origin of the
   * ancestor node's local coordinate frame.
   * (scenery-internal)
   */
  public getOriginBounds(): Bounds2 {
    return this.getCellBounds().shiftedXY( -this.proxy.x, -this.proxy.y );
  }

  /**
   * The current bounds of the cell (with margins included)
   * (scenery-internal)
   */
  public getCellBounds(): Bounds2 {
    return this.proxy.bounds.withOffsets(
      this.effectiveLeftMargin,
      this.effectiveTopMargin,
      this.effectiveRightMargin,
      this.effectiveBottomMargin
    );
  }

  public override dispose(): void {
    // Unset the specified preferred sizes that were set by our layout (when we're removed)
    Orientation.enumeration.values.forEach( orientation => {
      if ( this.preferredSizeSet.get( orientation ) ) {
        this.unsetPreferredSize( orientation );
      }
    } );

    super.dispose();
  }

  public static createHelperNode<Cell extends MarginLayoutCell>( cells: Cell[], layoutBounds: Bounds2, cellToText: ( cell: Cell ) => string ): Node {
    const container = new Node();
    const lineWidth = 0.4;

    const availableCellsShape = Shape.union( cells.map( cell => Shape.bounds( cell.lastAvailableBounds ) ) );
    const usedCellsShape = Shape.union( cells.map( cell => Shape.bounds( cell.lastUsedBounds ) ) );
    const usedContentShape = Shape.union( cells.map( cell => Shape.bounds( cell.proxy.bounds ) ) );
    const spacingShape = Shape.bounds( layoutBounds ).shapeDifference( availableCellsShape );
    const emptyShape = availableCellsShape.shapeDifference( usedCellsShape );
    const marginShape = usedCellsShape.shapeDifference( usedContentShape );

    const createLabeledTexture = ( label: string, foreground: TColor, background: TColor ) => {
      const text = new Text( label, {
        font: new Font( { size: 6, family: 'monospace' } ),
        fill: foreground
      } );
      const rectangle = Rectangle.bounds( text.bounds, {
        fill: background,
        children: [ text ]
      } );
      return new NodePattern(
        rectangle,
        4,
        Math.floor( rectangle.left ),
        Math.ceil( rectangle.top + 1 ),
        Math.floor( rectangle.width ),
        Math.floor( rectangle.height - 2 ),
        Matrix3.rotation2( -Math.PI / 4 )
      );
    };

    container.addChild( new Path( spacingShape, {
      fill: createLabeledTexture( 'spacing', '#000', '#fff' ),
      opacity: 0.6
    } ) );
    container.addChild( new Path( emptyShape, {
      fill: createLabeledTexture( 'empty', '#aaa', '#000' ),
      opacity: 0.6
    } ) );
    container.addChild( new Path( marginShape, {
      fill: createLabeledTexture( 'margin', '#600', '#f00' ),
      opacity: 0.6
    } ) );

    container.addChild( Rectangle.bounds( layoutBounds, {
      stroke: 'white',
      lineDash: [ 2, 2 ],
      lineDashOffset: 2,
      lineWidth: lineWidth
    } ) );
    container.addChild( Rectangle.bounds( layoutBounds, {
      stroke: 'black',
      lineDash: [ 2, 2 ],
      lineWidth: lineWidth
    } ) );

    cells.forEach( cell => {
      container.addChild( Rectangle.bounds( cell.getCellBounds(), {
        stroke: 'rgba(0,255,0,1)',
        lineWidth: lineWidth
      } ) );
    } );

    cells.forEach( cell => {
      container.addChild( Rectangle.bounds( cell.proxy.bounds, {
        stroke: 'rgba(255,0,0,1)',
        lineWidth: lineWidth
      } ) );
    } );

    cells.forEach( cell => {
      const bounds = cell.getCellBounds();

      const hoverListener = new PressListener( {
        tandem: Tandem.OPT_OUT
      } );
      container.addChild( Rectangle.bounds( bounds, {
        inputListeners: [ hoverListener ]
      } ) );

      let str = cellToText( cell );

      if ( cell.effectiveLeftMargin ) {
        str += `leftMargin: ${cell.effectiveLeftMargin}\n`;
      }
      if ( cell.effectiveRightMargin ) {
        str += `rightMargin: ${cell.effectiveRightMargin}\n`;
      }
      if ( cell.effectiveTopMargin ) {
        str += `topMargin: ${cell.effectiveTopMargin}\n`;
      }
      if ( cell.effectiveBottomMargin ) {
        str += `bottomMargin: ${cell.effectiveBottomMargin}\n`;
      }
      if ( cell.effectiveMinContentWidth ) {
        str += `minContentWidth: ${cell.effectiveMinContentWidth}\n`;
      }
      if ( cell.effectiveMinContentHeight ) {
        str += `minContentHeight: ${cell.effectiveMinContentHeight}\n`;
      }
      if ( cell.effectiveMaxContentWidth ) {
        str += `maxContentWidth: ${cell.effectiveMaxContentWidth}\n`;
      }
      if ( cell.effectiveMaxContentHeight ) {
        str += `maxContentHeight: ${cell.effectiveMaxContentHeight}\n`;
      }
      str += `layoutOptions: ${JSON.stringify( cell.node.layoutOptions, null, 2 ).replace( / /g, '&nbsp;' )}\n`;

      const hoverText = new RichText( str.trim().replace( /\n/g, '<br>' ), {
        font: new Font( { size: 12 } )
      } );
      const hoverNode = Rectangle.bounds( hoverText.bounds.dilated( 3 ), {
        fill: 'rgba(255,255,255,0.8)',
        children: [ hoverText ],
        leftTop: bounds.leftTop
      } );
      container.addChild( hoverNode );
      hoverListener.isOverProperty.link( isOver => {
        hoverNode.visible = isOver;
      } );
    } );

    return container;
  }
}

scenery.register( 'MarginLayoutCell', MarginLayoutCell );
