// Copyright 2021-2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import memoize from '../../../phet-core/js/memoize.js';
import mutate from '../../../phet-core/js/mutate.js';
import { HorizontalLayoutAlign, HorizontalLayoutAlignValues, LayoutAlign, scenery, VerticalLayoutAlign, VerticalLayoutAlignValues } from '../imports.js';

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
  // NOTE: 'origin' aligns will only apply to cells that are 1 grid line in that orientation (width/height)
  xAlign?: HorizontalLayoutAlign | null;
  yAlign?: VerticalLayoutAlign | null;
  stretch?: boolean;
  xStretch?: number | null;
  yStretch?: number | null;
  grow?: number | null;
  xGrow?: number | null;
  yGrow?: number | null;
  margin?: number | null;
  xMargin?: number | null;
  yMargin?: number | null;
  leftMargin?: number | null;
  rightMargin?: number | null;
  topMargin?: number | null;
  bottomMargin?: number | null;
  minContentWidth?: number | null;
  minContentHeight?: number | null;
  maxContentWidth?: number | null;
  maxContentHeight?: number | null;
};

const GridConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class extends type {

    _xAlign: LayoutAlign | null;
    _yAlign: LayoutAlign | null;
    _xStretch: boolean | null;
    _yStretch: boolean | null;
    _leftMargin: number | null;
    _rightMargin: number | null;
    _topMargin: number | null;
    _bottomMargin: number | null;
    _xGrow: number | null;
    _yGrow: number | null;
    _minContentWidth: number | null;
    _minContentHeight: number | null;
    _maxContentWidth: number | null;
    _maxContentHeight: number | null;

    readonly changedEmitter: TinyEmitter;

    constructor( ...args: any[] ) {
      super( ...args );

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

      this.changedEmitter = new TinyEmitter();
    }

    mutateConfigurable( options?: GridConfigurableOptions ): void {
      mutate( this, GRID_CONFIGURABLE_OPTION_KEYS, options );
    }

    setConfigToBaseDefault(): void {
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

    get xAlign(): HorizontalLayoutAlign | null {
      const result = this._xAlign === null ? null : this._xAlign.horizontal;

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    set xAlign( value: HorizontalLayoutAlign | null ) {
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

    get yAlign(): VerticalLayoutAlign | null {
      const result = this._yAlign === null ? null : this._yAlign.vertical;

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    set yAlign( value: VerticalLayoutAlign | null ) {
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
      assert && assert( this._xGrow === this._yGrow );

      return this._xGrow;
    }

    set grow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value || this._yGrow !== value ) {
        this._xGrow = value;
        this._yGrow = value;

        this.changedEmitter.emit();
      }
    }

    get xGrow(): number | null {
      return this._xGrow;
    }

    set xGrow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value ) {
        this._xGrow = value;

        this.changedEmitter.emit();
      }
    }

    get yGrow(): number | null {
      return this._yGrow;
    }

    set yGrow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._yGrow !== value ) {
        this._yGrow = value;

        this.changedEmitter.emit();
      }
    }

    get stretch(): boolean | null {
      assert && assert( this._xStretch === this._yStretch );

      return this._xStretch;
    }

    set stretch( value: boolean | null ) {
      assert && assert( value === null || typeof value === 'boolean' );

      if ( this._xStretch !== value || this._yStretch !== value ) {
        this._xStretch = value;
        this._yStretch = value;

        this.changedEmitter.emit();
      }
    }

    get xStretch(): boolean | null {
      return this._xStretch;
    }

    set xStretch( value: boolean | null ) {
      assert && assert( value === null || typeof value === 'boolean' );

      if ( this._xStretch !== value ) {
        this._xStretch = value;

        this.changedEmitter.emit();
      }
    }

    get yStretch(): boolean | null {
      return this._yStretch;
    }

    set yStretch( value: boolean | null ) {
      assert && assert( value === null || typeof value === 'boolean' );

      if ( this._yStretch !== value ) {
        this._yStretch = value;

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

scenery.register( 'GridConfigurable', GridConfigurable );
export default GridConfigurable;
export { GRID_CONFIGURABLE_OPTION_KEYS };
