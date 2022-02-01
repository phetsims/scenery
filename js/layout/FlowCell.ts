// Copyright 2021-2022, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import { scenery, FlowConfigurable, FlowConstraint, Node, FlowConfigurableOptions, FlowConfigurableAlign, isWidthSizable, isHeightSizable, WidthSizableNode, HeightSizableNode } from '../imports.js';

class FlowCell extends FlowConfigurable( Object ) {

  private _constraint: FlowConstraint;
  private _node: Node;
  public _pendingSize: number; // scenery-internal
  private layoutOptionsListener: () => void;

  constructor( constraint: FlowConstraint, node: Node, options?: FlowConfigurableOptions ) {
    super();

    this._constraint = constraint;
    this._node = node;
    this._pendingSize = 0;

    this.setOptions( options );

    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );
    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  get effectiveAlign(): FlowConfigurableAlign {
    return this._align !== null ? this._align : this._constraint._align!;
  }

  get effectiveLeftMargin(): number {
    return this._leftMargin !== null ? this._leftMargin : this._constraint._leftMargin!;
  }


  get effectiveRightMargin(): number {
    return this._rightMargin !== null ? this._rightMargin : this._constraint._rightMargin!;
  }


  get effectiveTopMargin(): number {
    return this._topMargin !== null ? this._topMargin : this._constraint._topMargin!;
  }


  get effectiveBottomMargin(): number {
    return this._bottomMargin !== null ? this._bottomMargin : this._constraint._bottomMargin!;
  }


  get effectiveGrow(): number {
    return this._grow !== null ? this._grow : this._constraint._grow!;
  }

  get effectiveMinContentWidth(): number | null {
    return this._minContentWidth !== null ? this._minContentWidth : this._constraint._minContentWidth;
  }

  get effectiveMinContentHeight(): number | null {
    return this._minContentHeight !== null ? this._minContentHeight : this._constraint._minContentHeight;
  }

  get effectiveMaxContentWidth(): number | null {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._constraint._maxContentWidth;
  }

  get effectiveMaxContentHeight(): number | null {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._constraint._maxContentHeight;
  }

  private onLayoutOptionsChange() {
    if ( this.node.layoutOptions ) {
      this.setOptions( this.node.layoutOptions as FlowConfigurableOptions );
    }
  }

  private setOptions( options?: FlowConfigurableOptions ) {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
  }

  get node(): Node {
    return this._node;
  }

  getMinimumSize( orientation: Orientation ): number {
    if ( orientation === Orientation.HORIZONTAL ) {
      return this.effectiveLeftMargin +
             Math.max( isWidthSizable( this.node ) ? this.node.minimumWidth || 0 : this.node.width, this.effectiveMinContentWidth || 0 ) +
             this.effectiveRightMargin;
    }
    else {
      return this.effectiveTopMargin +
             Math.max( isHeightSizable( this.node ) ? this.node.minimumHeight || 0 : this.node.height, this.effectiveMinContentHeight || 0 ) +
             this.effectiveBottomMargin;
    }
  }

  getMaximumSize( orientation: Orientation ): number {
    const isSizable = orientation === Orientation.HORIZONTAL ? isWidthSizable( this.node ) : isHeightSizable( this.node );

    if ( orientation === Orientation.HORIZONTAL ) {
      return this.effectiveLeftMargin +
             Math.min( isSizable ? Number.POSITIVE_INFINITY : this.node.width, this.effectiveMaxContentWidth || Number.POSITIVE_INFINITY ) +
             this.effectiveRightMargin;
    }
    else {
      return this.effectiveTopMargin +
             Math.min( isSizable ? Number.POSITIVE_INFINITY : this.node.height, this.effectiveMaxContentHeight || Number.POSITIVE_INFINITY ) +
             this.effectiveBottomMargin;
    }
  }

  attemptPreferredSize( orientation: Orientation, value: number ) {
    if ( orientation === Orientation.HORIZONTAL ? isWidthSizable( this.node ) : isHeightSizable( this.node ) ) {
      const minimumSize = this.getMinimumSize( orientation );
      const maximumSize = this.getMaximumSize( orientation );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      if ( orientation === Orientation.HORIZONTAL ) {
        ( this.node as WidthSizableNode ).preferredWidth = value - this.effectiveLeftMargin - this.effectiveRightMargin;
      }
      else {
        ( this.node as HeightSizableNode ).preferredHeight = value - this.effectiveTopMargin - this.effectiveBottomMargin;
      }
      // TODO: warnings if those preferred sizes weren't reached?
    }
  }

  positionStart( orientation: Orientation, value: number ) {
    // TODO: coordinate transform handling, to our ancestorNode!!!!!
    if ( orientation === Orientation.HORIZONTAL ) {
      const left = this.effectiveLeftMargin + value;

      if ( Math.abs( this.node.left - left ) > 1e-9 ) {
        this.node.left = left;
      }
    }
    else {
      const top = this.effectiveTopMargin + value;

      if ( Math.abs( this.node.top - top ) > 1e-9 ) {
        this.node.top = top;
      }
    }
  }

  positionOrigin( orientation: Orientation, value: number ) {
    if ( orientation === Orientation.HORIZONTAL ) {
      if ( Math.abs( this.node.x - value ) > 1e-9 ) {
        this.node.x = value;
      }
    }
    else {
      if ( Math.abs( this.node.y - value ) > 1e-9 ) {
        this.node.y = value;
      }
    }
  }

  getCellBounds(): Bounds2 {
    return this.node.bounds.withOffsets(
      this.effectiveLeftMargin,
      this.effectiveTopMargin,
      this.effectiveRightMargin,
      this.effectiveBottomMargin
     );
  }

  /**
   * Releases references
   */
  dispose() {
    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'FlowCell', FlowCell );
export default FlowCell;