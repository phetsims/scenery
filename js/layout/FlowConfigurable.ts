// Copyright 2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

// Disable for the whole file
/* eslint-disable no-protected-jsdoc */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import memoize from '../../../phet-core/js/memoize.js';
import mutate from '../../../phet-core/js/mutate.js';
import { HorizontalLayoutAlign, LayoutAlign, LayoutOrientation, scenery, VerticalLayoutAlign } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

const FLOW_CONFIGURABLE_OPTION_KEYS = [
  'orientation',
  'align',
  'stretch',
  'grow',
  'margin',
  'xMargin',
  'yMargin',
  'leftMargin',
  'rightMargin',
  'topMargin',
  'bottomMargin',
  'minContentWidth',
  'minContentHeight',
  'maxContentWidth',
  'maxContentHeight'
];

export type FlowConfigurableOptions = {
  // The main orientation of the layout that takes place. Items will be spaced out in this orientation (e.g. if it's
  // 'vertical', the y-values of the components will be adjusted to space them out); this is known as the "primary"
  // dimension. Items will be aligned/stretched in the opposite orientation (e.g. if it's 'vertical', the x-values of
  // the components will be adjusted by align and stretch); this is known as the "secondary" or "opposite" dimension.
  orientation?: LayoutOrientation | null;

  // Adjusts the position of elements in the "opposite" dimension, either to a specific side, the center, or so that all
  // the origins of items are aligned (similar to x=0 for a 'vertical' orientation).
  align?: HorizontalLayoutAlign | VerticalLayoutAlign | null;

  // Controls whether elements will attempt to expand in the "opposite" dimension to take up the full size of the
  // largest layout element.
  stretch?: boolean;

  // Controls whether elements will attempt to expand in the "primary" dimension. Elements will expand proportionally
  // based on the total grow sum (and will not expand at all if the grow is zero).
  grow?: number | null;

  // Adds extra space for each cell in the layout (margin controls all 4 sides, xMargin controls left/right, yMargin
  // controls top/bottom).
  margin?: number | null;
  xMargin?: number | null;
  yMargin?: number | null;
  leftMargin?: number | null;
  rightMargin?: number | null;
  topMargin?: number | null;
  bottomMargin?: number | null;

  // Forces size minimums and maximums on the cells (which includes the margins).
  // NOTE: For these, the nullable portion is actually part of the possible "value"
  minContentWidth?: number | null;
  minContentHeight?: number | null;
  maxContentWidth?: number | null;
  maxContentHeight?: number | null;
};

const FlowConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class extends type {

    _orientation: Orientation;

    _align: LayoutAlign | null;
    _stretch: boolean | null;
    _leftMargin: number | null;
    _rightMargin: number | null;
    _topMargin: number | null;
    _bottomMargin: number | null;
    _grow: number | null;
    _minContentWidth: number | null;
    _minContentHeight: number | null;
    _maxContentWidth: number | null;
    _maxContentHeight: number | null;

    readonly changedEmitter: TinyEmitter;
    readonly orientationChangedEmitter: TinyEmitter;

    constructor( ...args: any[] ) {
      super( ...args );

      this._orientation = Orientation.HORIZONTAL;

      this._align = null;
      this._stretch = null;
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter = new TinyEmitter();
      this.orientationChangedEmitter = new TinyEmitter();
    }

    mutateConfigurable( options?: FlowConfigurableOptions ): void {
      mutate( this, FLOW_CONFIGURABLE_OPTION_KEYS, options );
    }

    setConfigToBaseDefault(): void {
      this._align = LayoutAlign.CENTER;
      this._stretch = false;
      this._leftMargin = 0;
      this._rightMargin = 0;
      this._topMargin = 0;
      this._bottomMargin = 0;
      this._grow = 0;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter.emit();
    }

    /**
     * Resets values to their original state
     */
    setConfigToInherit(): void {
      this._align = null;
      this._stretch = null;
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter.emit();
    }

    get orientation(): LayoutOrientation {
      return this._orientation === Orientation.HORIZONTAL ? 'horizontal' : 'vertical';
    }

    set orientation( value: LayoutOrientation ) {
      assert && assert( value === 'horizontal' || value === 'vertical' );

      const enumOrientation = value === 'horizontal' ? Orientation.HORIZONTAL : Orientation.VERTICAL;

      if ( this._orientation !== enumOrientation ) {
        this._orientation = enumOrientation;

        this.orientationChangedEmitter.emit();
        this.changedEmitter.emit();
      }
    }

    get align(): HorizontalLayoutAlign | VerticalLayoutAlign | null {
      const result = LayoutAlign.internalToAlign( this._orientation, this._align );

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    set align( value: HorizontalLayoutAlign | VerticalLayoutAlign | null ) {
      assert && assert( LayoutAlign.getAllowedAligns( this._orientation.opposite ).includes( value ),
        `align ${value} not supported, with the orientation ${this._orientation}, the valid values are ${LayoutAlign.getAllowedAligns( this._orientation.opposite )}` );

      // remapping align values to an independent set, so they aren't orientation-dependent
      const mappedValue = LayoutAlign.alignToInternal( this._orientation.opposite, value );

      assert && assert( mappedValue === null || mappedValue instanceof LayoutAlign );

      if ( this._align !== mappedValue ) {
        this._align = mappedValue;

        this.changedEmitter.emit();
      }
    }

    get stretch(): boolean | null {
      return this._stretch;
    }

    set stretch( value: boolean | null ) {
      if ( this._stretch !== value ) {
        this._stretch = value;

        this.changedEmitter.emit();
      }
    }

    get leftMargin(): number | null {
      return this._leftMargin;
    }

    set leftMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value ) {
        this._leftMargin = value;

        this.changedEmitter.emit();
      }
    }

    get rightMargin(): number | null {
      return this._rightMargin;
    }

    set rightMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._rightMargin !== value ) {
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    get topMargin(): number | null {
      return this._topMargin;
    }

    set topMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value ) {
        this._topMargin = value;

        this.changedEmitter.emit();
      }
    }

    get bottomMargin(): number | null {
      return this._bottomMargin;
    }

    set bottomMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._bottomMargin !== value ) {
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    get grow(): number | null {
      return this._grow;
    }

    set grow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._grow !== value ) {
        this._grow = value;

        this.changedEmitter.emit();
      }
    }

    get xMargin(): number | null {
      assert && assert( this._leftMargin === this._rightMargin );

      return this._leftMargin;
    }

    set xMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    get yMargin(): number | null {
      assert && assert( this._topMargin === this._bottomMargin );

      return this._topMargin;
    }

    set yMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value || this._bottomMargin !== value ) {
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    get margin(): number | null {
      assert && assert(
      this._leftMargin === this._rightMargin &&
      this._leftMargin === this._topMargin &&
      this._leftMargin === this._bottomMargin
      );

      return this._topMargin;
    }

    set margin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value || this._topMargin !== value || this._bottomMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    get minContentWidth(): number | null {
      return this._minContentWidth;
    }

    set minContentWidth( value: number | null ) {
      if ( this._minContentWidth !== value ) {
        this._minContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    get minContentHeight(): number | null {
      return this._minContentHeight;
    }

    set minContentHeight( value: number | null ) {
      if ( this._minContentHeight !== value ) {
        this._minContentHeight = value;

        this.changedEmitter.emit();
      }
    }

    get maxContentWidth(): number | null {
      return this._maxContentWidth;
    }

    set maxContentWidth( value: number | null ) {
      if ( this._maxContentWidth !== value ) {
        this._maxContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    get maxContentHeight(): number | null {
      return this._maxContentHeight;
    }

    set maxContentHeight( value: number | null ) {
      if ( this._maxContentHeight !== value ) {
        this._maxContentHeight = value;

        this.changedEmitter.emit();
      }
    }
  };
} );

scenery.register( 'FlowConfigurable', FlowConfigurable );
export default FlowConfigurable;
export { FLOW_CONFIGURABLE_OPTION_KEYS };
