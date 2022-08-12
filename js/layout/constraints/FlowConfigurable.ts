// Copyright 2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell. `null` for values usually means "inherit from the default".
 *
 * Handles a lot of conversion from internal Enumeration values (for performance) and external string representations.
 * This is done primarily for performance and that style of internal enumeration pattern. If string comparisons are
 * faster, that could be used instead.
 *
 * NOTE: Internal non-string representations are also orientation-agnostic
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import memoize from '../../../../phet-core/js/memoize.js';
import mutate from '../../../../phet-core/js/mutate.js';
import { HorizontalLayoutAlign, LayoutAlign, LayoutOrientation, scenery, VerticalLayoutAlign } from '../../imports.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import assertMutuallyExclusiveOptions from '../../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import WithoutNull from '../../../../phet-core/js/types/WithoutNull.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import TEmitter from '../../../../axon/js/TEmitter.js';

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
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-orientation
  orientation?: LayoutOrientation | null;

  // Adjusts the position of elements in the "opposite" dimension, either to a specific side, the center, or so that all
  // the origins of items are aligned (similar to x=0 for a 'vertical' orientation).
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-align
  align?: HorizontalLayoutAlign | VerticalLayoutAlign | null;

  // Controls whether elements will attempt to expand in the "opposite" dimension to take up the full size of the
  // largest layout element.
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-stretch
  stretch?: boolean;

  // Controls whether elements will attempt to expand in the "primary" dimension. Elements will expand proportionally
  // based on the total grow sum (and will not expand at all if the grow is zero).
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-grow
  grow?: number | null;

  // Adds extra space for each cell in the layout (margin controls all 4 sides, xMargin controls left/right, yMargin
  // controls top/bottom).
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-margins
  margin?: number | null;
  xMargin?: number | null;
  yMargin?: number | null;
  leftMargin?: number | null;
  rightMargin?: number | null;
  topMargin?: number | null;
  bottomMargin?: number | null;

  // Forces size minimums and maximums on the cells (which does not include the margins).
  // NOTE: For these, the nullable portion is actually part of the possible "value"
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-minContent and
  // https://phetsims.github.io/scenery/doc/layout#FlowBox-maxContent
  minContentWidth?: number | null;
  minContentHeight?: number | null;
  maxContentWidth?: number | null;
  maxContentHeight?: number | null;
};

// We remove the null values for the values that won't actually take null
export type ExternalFlowConfigurableOptions = WithoutNull<FlowConfigurableOptions, Exclude<keyof FlowConfigurableOptions, 'minContentWidth' | 'minContentHeight' | 'maxContentWidth' | 'maxContentHeight'>>;

const FlowConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class FlowConfirableMixin extends type {

    protected _orientation: Orientation = Orientation.HORIZONTAL;

    // (scenery-internal)
    public _align: LayoutAlign | null = null;
    public _stretch: boolean | null = null;
    public _leftMargin: number | null = null;
    public _rightMargin: number | null = null;
    public _topMargin: number | null = null;
    public _bottomMargin: number | null = null;
    public _grow: number | null = null;
    public _minContentWidth: number | null = null;
    public _minContentHeight: number | null = null;
    public _maxContentWidth: number | null = null;
    public _maxContentHeight: number | null = null;

    public readonly changedEmitter: TEmitter = new TinyEmitter();
    public readonly orientationChangedEmitter: TEmitter = new TinyEmitter();

    /**
     * (scenery-internal)
     */
    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );
    }

    /**
     * (scenery-internal)
     */
    public mutateConfigurable( options?: FlowConfigurableOptions ): void {
      assertMutuallyExclusiveOptions( options, [ 'margin' ], [ 'xMargin', 'yMargin' ] );
      assertMutuallyExclusiveOptions( options, [ 'xMargin' ], [ 'leftMargin', 'rightMargin' ] );
      assertMutuallyExclusiveOptions( options, [ 'yMargin' ], [ 'topMargin', 'bottomMargin' ] );

      mutate( this, FLOW_CONFIGURABLE_OPTION_KEYS, options );
    }

    /**
     * Resets values to the "base" state
     * (scenery-internal)
     */
    public setConfigToBaseDefault(): void {
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
     * (scenery-internal)
     */
    public setConfigToInherit(): void {
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

    /**
     * (scenery-internal)
     */
    public get orientation(): LayoutOrientation {
      return this._orientation === Orientation.HORIZONTAL ? 'horizontal' : 'vertical';
    }

    /**
     * (scenery-internal)
     */
    public set orientation( value: LayoutOrientation ) {
      assert && assert( value === 'horizontal' || value === 'vertical' );

      const enumOrientation = value === 'horizontal' ? Orientation.HORIZONTAL : Orientation.VERTICAL;

      if ( this._orientation !== enumOrientation ) {
        this._orientation = enumOrientation;

        this.orientationChangedEmitter.emit();
        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get align(): HorizontalLayoutAlign | VerticalLayoutAlign | null {
      const result = LayoutAlign.internalToAlign( this._orientation, this._align );

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    /**
     * (scenery-internal)
     */
    public set align( value: HorizontalLayoutAlign | VerticalLayoutAlign | null ) {
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

    /**
     * (scenery-internal)
     */
    public get stretch(): boolean | null {
      return this._stretch;
    }

    /**
     * (scenery-internal)
     */
    public set stretch( value: boolean | null ) {
      if ( this._stretch !== value ) {
        this._stretch = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get leftMargin(): number | null {
      return this._leftMargin;
    }

    /**
     * (scenery-internal)
     */
    public set leftMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value ) {
        this._leftMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get rightMargin(): number | null {
      return this._rightMargin;
    }

    /**
     * (scenery-internal)
     */
    public set rightMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._rightMargin !== value ) {
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get topMargin(): number | null {
      return this._topMargin;
    }

    /**
     * (scenery-internal)
     */
    public set topMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value ) {
        this._topMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get bottomMargin(): number | null {
      return this._bottomMargin;
    }

    /**
     * (scenery-internal)
     */
    public set bottomMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._bottomMargin !== value ) {
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get grow(): number | null {
      return this._grow;
    }

    /**
     * (scenery-internal)
     */
    public set grow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._grow !== value ) {
        this._grow = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get xMargin(): number | null {
      assert && assert( this._leftMargin === this._rightMargin );

      return this._leftMargin;
    }

    /**
     * (scenery-internal)
     */
    public set xMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get yMargin(): number | null {
      assert && assert( this._topMargin === this._bottomMargin );

      return this._topMargin;
    }

    /**
     * (scenery-internal)
     */
    public set yMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value || this._bottomMargin !== value ) {
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get margin(): number | null {
      assert && assert(
      this._leftMargin === this._rightMargin &&
      this._leftMargin === this._topMargin &&
      this._leftMargin === this._bottomMargin
      );

      return this._topMargin;
    }

    /**
     * (scenery-internal)
     */
    public set margin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value || this._topMargin !== value || this._bottomMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get minContentWidth(): number | null {
      return this._minContentWidth;
    }

    /**
     * (scenery-internal)
     */
    public set minContentWidth( value: number | null ) {
      if ( this._minContentWidth !== value ) {
        this._minContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get minContentHeight(): number | null {
      return this._minContentHeight;
    }

    /**
     * (scenery-internal)
     */
    public set minContentHeight( value: number | null ) {
      if ( this._minContentHeight !== value ) {
        this._minContentHeight = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get maxContentWidth(): number | null {
      return this._maxContentWidth;
    }

    /**
     * (scenery-internal)
     */
    public set maxContentWidth( value: number | null ) {
      if ( this._maxContentWidth !== value ) {
        this._maxContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get maxContentHeight(): number | null {
      return this._maxContentHeight;
    }

    /**
     * (scenery-internal)
     */
    public set maxContentHeight( value: number | null ) {
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
