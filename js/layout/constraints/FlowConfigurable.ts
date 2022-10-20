// Copyright 2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell. `null` for values usually means "inherit from the default".
 *
 * Handles a lot of conversion from internal Enumeration values (for performance) and external string representations.
 * This is done primarily for performance and that style of internal enumeration pattern. If string comparisons are
 * faster, that could be used instead.
 *
 * NOTE: Internal non-string representations are also orientation-agnostic - thus "left" and "top" map to the same
 * "start" internally, and thus the external value will appear to "switch" depending on the orientation.
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
import Orientation from '../../../../phet-core/js/Orientation.js';
import memoize from '../../../../phet-core/js/memoize.js';
import mutate from '../../../../phet-core/js/mutate.js';
import { HorizontalLayoutAlign, LayoutAlign, LayoutOrientation, MARGIN_LAYOUT_CONFIGURABLE_OPTION_KEYS, MarginLayoutConfigurable, MarginLayoutConfigurableOptions, scenery, VerticalLayoutAlign } from '../../imports.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import WithoutNull from '../../../../phet-core/js/types/WithoutNull.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import TEmitter from '../../../../axon/js/TEmitter.js';

const FLOW_CONFIGURABLE_OPTION_KEYS = [
  'orientation',
  'align',
  'stretch',
  'grow'
].concat( MARGIN_LAYOUT_CONFIGURABLE_OPTION_KEYS );

type SelfOptions = {
  // The main orientation of the layout that takes place. Items will be spaced out in this orientation (e.g. if it's
  // 'vertical', the y-values of the components will be adjusted to space them out); this is known as the "primary"
  // axis. Items will be aligned/stretched in the opposite orientation (e.g. if it's 'vertical', the x-values of
  // the components will be adjusted by align and stretch); this is known as the "secondary" or "opposite" axis.
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-orientation
  orientation?: LayoutOrientation | null;

  // Adjusts the position of elements in the "opposite" dimension, either to a specific side, the center, or so that all
  // the origins of items are aligned (similar to x=0 for a 'vertical' orientation).
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-align
  align?: HorizontalLayoutAlign | VerticalLayoutAlign | null;

  // Controls whether elements will attempt to expand along the "opposite" axis to take up the full size of the
  // largest layout element.
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-stretch
  stretch?: boolean;

  // Controls whether elements will attempt to expand along the "primary" axis. Elements will expand proportionally
  // based on the total grow sum (and will not expand at all if the grow is zero).
  // See https://phetsims.github.io/scenery/doc/layout#FlowBox-grow
  grow?: number | null;
};

export type FlowConfigurableOptions = SelfOptions & MarginLayoutConfigurableOptions;

// We remove the null values for the values that won't actually take null
export type ExternalFlowConfigurableOptions = WithoutNull<FlowConfigurableOptions, Exclude<keyof FlowConfigurableOptions, 'minContentWidth' | 'minContentHeight' | 'maxContentWidth' | 'maxContentHeight'>>;

// (scenery-internal)
const FlowConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class FlowConfigurableMixin extends MarginLayoutConfigurable( type ) {

    protected _orientation: Orientation = Orientation.HORIZONTAL;

    // (scenery-internal)
    public _align: LayoutAlign | null = null;
    public _stretch: boolean | null = null;
    public _grow: number | null = null;

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
    public override mutateConfigurable( options?: FlowConfigurableOptions ): void {
      super.mutateConfigurable( options );

      mutate( this, FLOW_CONFIGURABLE_OPTION_KEYS, options );
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
      this._align = LayoutAlign.CENTER;
      this._stretch = false;
      this._grow = 0;

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
      this._align = null;
      this._stretch = null;
      this._grow = null;

      super.setConfigToInherit();
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
  };
} );

scenery.register( 'FlowConfigurable', FlowConfigurable );
export default FlowConfigurable;
export { FLOW_CONFIGURABLE_OPTION_KEYS };
