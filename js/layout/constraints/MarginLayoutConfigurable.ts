// Copyright 2022, University of Colorado Boulder

/**
 * This combines the margin-cell related options common to FlowConfigurable and GridConfigurable
 * Parent mixin for flow/grid configurables (mixins for storing options that can affect each cell).
 * `null` for values usually means "inherit from the default".
 *
 * NOTE: This is mixed into both the constraint AND the cell, since we have two layers of options. The `null` meaning
 * "inherit from the default" is mainly used for the cells, so that if it's not specified in the cell, it will be
 * specified in the constraint (as non-null).
 *
 * NOTE: This is a mixin meant to be used internally only by Scenery (for the constraint and cell), and should not be
 * used by outside code.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import memoize from '../../../../phet-core/js/memoize.js';
import { scenery } from '../../imports.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import assertMutuallyExclusiveOptions from '../../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import WithoutNull from '../../../../phet-core/js/types/WithoutNull.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import TEmitter from '../../../../axon/js/TEmitter.js';

const MARGIN_LAYOUT_CONFIGURABLE_OPTION_KEYS = [
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

export type MarginLayoutConfigurableOptions = {
  // Adds extra space for each cell in the layout (margin controls all 4 sides, xMargin controls left/right, yMargin
  // controls top/bottom).
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-margins
  // See https://phetsims.github.io/scenery/doc/layout#GridBox-margins
  // Margins will control how much extra space is FORCED around content within a cell's available area. These margins do
  // not collapse (each cell gets its own).
  margin?: number | null; // shortcut for left/right/top/bottom margins
  xMargin?: number | null; // shortcut for left/right margins
  yMargin?: number | null; // shortcut for top/bottom margins
  leftMargin?: number | null;
  rightMargin?: number | null;
  topMargin?: number | null;
  bottomMargin?: number | null;

  // Forces size minimums and maximums on the cells (which does not include the margins).
  // NOTE: For these, the nullable portion is actually part of the possible "value"
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-minContent and
  // https://phetsims.github.io/scenery/doc/layout#FlowBox-maxContent
  // See https://phetsims.github.io/scenery/doc/layout#GridBox-minContent and
  // https://phetsims.github.io/scenery/doc/layout#GridBox-maxContent
  minContentWidth?: number | null;
  minContentHeight?: number | null;
  maxContentWidth?: number | null;
  maxContentHeight?: number | null;
};

// We remove the null values for the values that won't actually take null
export type ExternalMarginLayoutConfigurableOptions = WithoutNull<MarginLayoutConfigurableOptions, Exclude<keyof MarginLayoutConfigurableOptions, 'minContentWidth' | 'minContentHeight' | 'maxContentWidth' | 'maxContentHeight'>>;

// (scenery-internal)
const MarginLayoutConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class MarginLayoutConfigurableMixin extends type {

    // (scenery-internal)
    public _leftMargin: number | null = null;
    public _rightMargin: number | null = null;
    public _topMargin: number | null = null;
    public _bottomMargin: number | null = null;
    public _minContentWidth: number | null = null;
    public _minContentHeight: number | null = null;
    public _maxContentWidth: number | null = null;
    public _maxContentHeight: number | null = null;

    public readonly changedEmitter: TEmitter = new TinyEmitter();

    /**
     * (scenery-internal)
     */
    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );
    }

    /**
     * (scenery-internal)
     */
    public mutateConfigurable( options?: MarginLayoutConfigurableOptions ): void {
      assertMutuallyExclusiveOptions( options, [ 'margin' ], [ 'xMargin', 'yMargin' ] );
      assertMutuallyExclusiveOptions( options, [ 'xMargin' ], [ 'leftMargin', 'rightMargin' ] );
      assertMutuallyExclusiveOptions( options, [ 'yMargin' ], [ 'topMargin', 'bottomMargin' ] );
    }

    /**
     * Resets values to the "base" state.
     *
     * This is the fallback state for a constraint where every value is defined and valid. If a cell does not have a
     * specific "overridden" value, or a constraint doesn't have an "overridden" value, then it will take the value
     * defined here.
     *
     * These should be the default values for constraints.
     *
     * NOTE: min/max content width/height are null here (since null is a valid default, and doesn't indicate an
     * "inherit" value like the other types).
     *
     * (scenery-internal)
     */
    public setConfigToBaseDefault(): void {
      this._leftMargin = 0;
      this._rightMargin = 0;
      this._topMargin = 0;
      this._bottomMargin = 0;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter.emit();
    }

    /**
     * Resets values to the "don't override anything, only inherit from the constraint" state
     *
     * These should be the default values for cells (e.g. "take all the behavior from the constraint, nothing is
     * overridden").
     *
     * (scenery-internal)
     */
    public setConfigToInherit(): void {
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter.emit();
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

scenery.register( 'MarginLayoutConfigurable', MarginLayoutConfigurable );
export default MarginLayoutConfigurable;
export { MARGIN_LAYOUT_CONFIGURABLE_OPTION_KEYS };
