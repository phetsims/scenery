// Copyright 2021-2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell.
 *
 * Handles a lot of conversion from internal Enumeration values (for performance) and external string representations.
 * This is done primarily for performance and that style of internal enumeration pattern. If string comparisons are
 * faster, that could be used instead.
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

import Constructor from '../../../../phet-core/js/types/Constructor.js';
import memoize from '../../../../phet-core/js/memoize.js';
import mutate from '../../../../phet-core/js/mutate.js';
import { HorizontalLayoutAlign, HorizontalLayoutAlignValues, LayoutAlign, MARGIN_LAYOUT_CONFIGURABLE_OPTION_KEYS, MarginLayoutConfigurable, MarginLayoutConfigurableOptions, scenery, VerticalLayoutAlign, VerticalLayoutAlignValues } from '../../imports.js';
import assertMutuallyExclusiveOptions from '../../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import WithoutNull from '../../../../phet-core/js/types/WithoutNull.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';

const GRID_CONFIGURABLE_OPTION_KEYS = [
  'xAlign',
  'yAlign',
  'stretch',
  'xStretch',
  'yStretch',
  'grow',
  'xGrow',
  'yGrow'
].concat( MARGIN_LAYOUT_CONFIGURABLE_OPTION_KEYS );

type SelfOptions = {
  // Alignments control how the content of a cell is positioned within that cell's available area (thus it only applies
  // if there is ADDITIONAL space, e.g. in a row/column with a larger item, or there is a preferred size on the GridBox.
  //
  // For 'origin', the x=0 or y=0 points of each item content will be aligned (vertically or horizontally). This is
  // particularly useful for Text, where the origin (y=0) is the baseline of the text, so that differently-sized texts
  // can have their baselines aligned, or other content can be aligned (e.g. a circle whose origin is at its center).
  //
  // NOTE: 'origin' aligns will only apply to cells that are 1 grid line in that orientation (width/height)
  xAlign?: HorizontalLayoutAlign | null;
  yAlign?: VerticalLayoutAlign | null;

  // Stretch will control whether a resizable component (mixes in WidthSizable/HeightSizable) will expand to fill the
  // available space within a cell's available area. Similarly to align, this only applies if there is additional space.
  stretch?: boolean; // shortcut for xStretch/yStretch
  xStretch?: boolean | null;
  yStretch?: boolean | null;

  // Grow will control how additional empty space (above the minimum sizes that the grid could take) will be
  // proportioned out to the rows and columns. Unlike stretch, this affects the size of the columns, and does not affect
  // the individual cells.
  grow?: number | null; // shortcut for xGrow/yGrow
  xGrow?: number | null;
  yGrow?: number | null;
};

export type GridConfigurableOptions = SelfOptions & MarginLayoutConfigurableOptions;

// We remove the null values for the values that won't actually take null
export type ExternalGridConfigurableOptions = WithoutNull<GridConfigurableOptions, Exclude<keyof GridConfigurableOptions, 'minContentWidth' | 'minContentHeight' | 'maxContentWidth' | 'maxContentHeight'>>;

// (scenery-internal)
const GridConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class GridConfigurableMixin extends MarginLayoutConfigurable( type ) {

    // (scenery-internal)
    public _xAlign: LayoutAlign | null = null;
    public _yAlign: LayoutAlign | null = null;
    public _xStretch: boolean | null = null;
    public _yStretch: boolean | null = null;
    public _xGrow: number | null = null;
    public _yGrow: number | null = null;

    /**
     * (scenery-internal)
     */
    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );
    }

    /**
     * (scenery-internal)
     */
    public override mutateConfigurable( options?: GridConfigurableOptions ): void {
      super.mutateConfigurable( options );

      assertMutuallyExclusiveOptions( options, [ 'stretch' ], [ 'xStretch', 'yStretch' ] );
      assertMutuallyExclusiveOptions( options, [ 'grow' ], [ 'xGrow', 'yGrow' ] );

      mutate( this, GRID_CONFIGURABLE_OPTION_KEYS, options );
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
     * (scenery-internal)
     */
    public override setConfigToBaseDefault(): void {
      this._xAlign = LayoutAlign.CENTER;
      this._yAlign = LayoutAlign.CENTER;
      this._xStretch = false;
      this._yStretch = false;
      this._xGrow = 0;
      this._yGrow = 0;

      super.setConfigToBaseDefault();
    }

    /**
     * Resets values to the "don't override anything, only inherit from the constraint" state
     *
     * These should be the default values for cells (e.g. "take all the behavior from the constraint, nothing is
     * overridden").
     *
     * (scenery-internal)
     */
    public override setConfigToInherit(): void {
      this._xAlign = null;
      this._yAlign = null;
      this._xStretch = null;
      this._yStretch = null;
      this._xGrow = null;
      this._yGrow = null;

      super.setConfigToInherit();
    }

    /**
     * (scenery-internal)
     */
    public get xAlign(): HorizontalLayoutAlign | null {
      const result = this._xAlign === null ? null : this._xAlign.horizontal;

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    /**
     * (scenery-internal)
     */
    public set xAlign( value: HorizontalLayoutAlign | null ) {
      assert && assert( value === null || HorizontalLayoutAlignValues.includes( value ),
        `align ${value} not supported, the valid values are ${HorizontalLayoutAlignValues} or null` );

      // remapping align values to an independent set, so they aren't orientation-dependent
      const mappedValue = LayoutAlign.horizontalAlignToInternal( value );

      assert && assert( mappedValue === null || mappedValue instanceof LayoutAlign );

      if ( this._xAlign !== mappedValue ) {
        this._xAlign = mappedValue;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get yAlign(): VerticalLayoutAlign | null {
      const result = this._yAlign === null ? null : this._yAlign.vertical;

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    /**
     * (scenery-internal)
     */
    public set yAlign( value: VerticalLayoutAlign | null ) {
      assert && assert( value === null || VerticalLayoutAlignValues.includes( value ),
        `align ${value} not supported, the valid values are ${VerticalLayoutAlignValues} or null` );

      // remapping align values to an independent set, so they aren't orientation-dependent
      const mappedValue = LayoutAlign.verticalAlignToInternal( value );

      assert && assert( mappedValue === null || mappedValue instanceof LayoutAlign );

      if ( this._yAlign !== mappedValue ) {
        this._yAlign = mappedValue;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get grow(): number | null {
      assert && assert( this._xGrow === this._yGrow );

      return this._xGrow;
    }

    /**
     * (scenery-internal)
     */
    public set grow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value || this._yGrow !== value ) {
        this._xGrow = value;
        this._yGrow = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get xGrow(): number | null {
      return this._xGrow;
    }

    /**
     * (scenery-internal)
     */
    public set xGrow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value ) {
        this._xGrow = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get yGrow(): number | null {
      return this._yGrow;
    }

    /**
     * (scenery-internal)
     */
    public set yGrow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._yGrow !== value ) {
        this._yGrow = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get stretch(): boolean | null {
      assert && assert( this._xStretch === this._yStretch );

      return this._xStretch;
    }

    /**
     * (scenery-internal)
     */
    public set stretch( value: boolean | null ) {
      assert && assert( value === null || typeof value === 'boolean' );

      if ( this._xStretch !== value || this._yStretch !== value ) {
        this._xStretch = value;
        this._yStretch = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get xStretch(): boolean | null {
      return this._xStretch;
    }

    /**
     * (scenery-internal)
     */
    public set xStretch( value: boolean | null ) {
      assert && assert( value === null || typeof value === 'boolean' );

      if ( this._xStretch !== value ) {
        this._xStretch = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * (scenery-internal)
     */
    public get yStretch(): boolean | null {
      return this._yStretch;
    }

    /**
     * (scenery-internal)
     */
    public set yStretch( value: boolean | null ) {
      assert && assert( value === null || typeof value === 'boolean' );

      if ( this._yStretch !== value ) {
        this._yStretch = value;

        this.changedEmitter.emit();
      }
    }
  };
} );

scenery.register( 'GridConfigurable', GridConfigurable );
export default GridConfigurable;
export { GRID_CONFIGURABLE_OPTION_KEYS };
