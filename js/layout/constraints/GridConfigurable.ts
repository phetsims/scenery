// Copyright 2021-2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell.
 *
 * Handles a lot of conversion from internal Enumeration values (for performance) and external string representations.
 * This is done primarily for performance and that style of internal enumeration pattern. If string comparisons are
 * faster, that could be used instead.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import memoize from '../../../../phet-core/js/memoize.js';
import mutate from '../../../../phet-core/js/mutate.js';
import { HorizontalLayoutAlign, HorizontalLayoutAlignValues, LayoutAlign, scenery, VerticalLayoutAlign, VerticalLayoutAlignValues } from '../../imports.js';
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
  'yGrow',
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

export type GridConfigurableOptions = {
  // Alignments control how the content of a cell is positioned within that cell's available area (thus it only applies
  // if there is ADDITIONAL space, e.g. in a row/column with a larger item, or there is a preferred size on the GridBox.
  // NOTE: 'origin' aligns will only apply to cells that are 1 grid line in that orientation (width/height)
  xAlign?: HorizontalLayoutAlign | null;
  yAlign?: VerticalLayoutAlign | null;

  // Stretch will control whether a resizable component (mixes in WidthSizable/HeightSizable) will expand to fill the
  // available space within a cell's available area. Similarly to align, this only applies if there is additional space.
  stretch?: boolean; // shortcut for xStretch/yStretch
  xStretch?: number | null;
  yStretch?: number | null;

  // Grow will control how additional empty space (above the minimum sizes that the grid could take) will be
  // proportioned out to the rows and columns. Unlike stretch, this affects the size of the columns, and does not affect
  // the individual cells.
  grow?: number | null; // shortcut for xGrow/yGrow
  xGrow?: number | null;
  yGrow?: number | null;

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
  minContentWidth?: number | null;
  minContentHeight?: number | null;
  maxContentWidth?: number | null;
  maxContentHeight?: number | null;
};

// We remove the null values for the values that won't actually take null
export type ExternalGridConfigurableOptions = WithoutNull<GridConfigurableOptions, Exclude<keyof GridConfigurableOptions, 'minContentWidth' | 'minContentHeight' | 'maxContentWidth' | 'maxContentHeight'>>;

const GridConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class GridConfigurableMixin extends type {

    // (scenery-internal)
    public _xAlign: LayoutAlign | null = null;
    public _yAlign: LayoutAlign | null = null;
    public _xStretch: boolean | null = null;
    public _yStretch: boolean | null = null;
    public _leftMargin: number | null = null;
    public _rightMargin: number | null = null;
    public _topMargin: number | null = null;
    public _bottomMargin: number | null = null;
    public _xGrow: number | null = null;
    public _yGrow: number | null = null;
    public _minContentWidth: number | null = null;
    public _minContentHeight: number | null = null;
    public _maxContentWidth: number | null = null;
    public _maxContentHeight: number | null = null;

    public readonly changedEmitter: TinyEmitter = new TinyEmitter<[]>();

    /**
     * (scenery-internal)
     */
    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );
    }

    /**
     * (scenery-internal)
     */
    public mutateConfigurable( options?: GridConfigurableOptions ): void {
      assertMutuallyExclusiveOptions( options, [ 'stretch' ], [ 'xStretch', 'yStretch' ] );
      assertMutuallyExclusiveOptions( options, [ 'grow' ], [ 'xGrow', 'yGrow' ] );
      assertMutuallyExclusiveOptions( options, [ 'margin' ], [ 'xMargin', 'yMargin' ] );
      assertMutuallyExclusiveOptions( options, [ 'xMargin' ], [ 'leftMargin', 'rightMargin' ] );
      assertMutuallyExclusiveOptions( options, [ 'yMargin' ], [ 'topMargin', 'bottomMargin' ] );

      mutate( this, GRID_CONFIGURABLE_OPTION_KEYS, options );
    }

    /**
     * (scenery-internal)
     */
    public setConfigToBaseDefault(): void {
      this._xAlign = LayoutAlign.CENTER;
      this._yAlign = LayoutAlign.CENTER;
      this._xStretch = false;
      this._yStretch = false;
      this._leftMargin = 0;
      this._rightMargin = 0;
      this._topMargin = 0;
      this._bottomMargin = 0;
      this._xGrow = 0;
      this._yGrow = 0;
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
    setConfigToInherit(): void {
      this._xAlign = null;
      this._yAlign = null;
      this._xStretch = null;
      this._yStretch = null;
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._xGrow = null;
      this._yGrow = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter.emit();
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

scenery.register( 'GridConfigurable', GridConfigurable );
export default GridConfigurable;
export { GRID_CONFIGURABLE_OPTION_KEYS };
