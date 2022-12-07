// Copyright 2021-2022, University of Colorado Boulder

/**
 * Trait for Nodes that support a standard fill and/or stroke (e.g. Text, Path and Path subtypes).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import ReadOnlyProperty from '../../../axon/js/ReadOnlyProperty.js';
import { LINE_STYLE_DEFAULT_OPTIONS, LineCap, LineJoin, LineStyles } from '../../../kite/js/imports.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import assertHasProperties from '../../../phet-core/js/assertHasProperties.js';
import inheritance from '../../../phet-core/js/inheritance.js';
import platform from '../../../phet-core/js/platform.js';
import memoize from '../../../phet-core/js/memoize.js';
import { CanvasContextWrapper, Color, Gradient, TPaint, TPaintableDrawable, LinearGradient, Node, Paint, PaintDef, Path, Pattern, RadialGradient, Renderer, scenery, Text } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import Vector2 from '../../../dot/js/Vector2.js';

const isSafari5 = platform.safari5;

const PAINTABLE_OPTION_KEYS = [
  'fill', // {PaintDef} - Sets the fill of this Node, see setFill() for documentation.
  'fillPickable', // {boolean} - Sets whether the filled area of the Node will be treated as 'inside'. See setFillPickable()
  'stroke', // {PaintDef} - Sets the stroke of this Node, see setStroke() for documentation.
  'strokePickable', // {boolean} - Sets whether the stroked area of the Node will be treated as 'inside'. See setStrokePickable()
  'lineWidth', // {number} - Sets the width of the stroked area, see setLineWidth for documentation.
  'lineCap', // {string} - Sets the shape of the stroked area at the start/end of the path, see setLineCap() for documentation.
  'lineJoin', // {string} - Sets the shape of the stroked area at joints, see setLineJoin() for documentation.
  'miterLimit', // {number} - Sets when lineJoin will switch from miter to bevel, see setMiterLimit() for documentation.
  'lineDash', // {Array.<number>} - Sets a line-dash pattern for the stroke, see setLineDash() for documentation
  'lineDashOffset', // {number} - Sets the offset of the line-dash from the start of the stroke, see setLineDashOffset()
  'cachedPaints' // {Array.<PaintDef>} - Sets which paints should be cached, even if not displayed. See setCachedPaints()
];

const DEFAULT_OPTIONS = {
  fill: null,
  fillPickable: true,
  stroke: null,
  strokePickable: false,

  // Not set initially, but they are the LineStyles defaults
  lineWidth: LINE_STYLE_DEFAULT_OPTIONS.lineWidth,
  lineCap: LINE_STYLE_DEFAULT_OPTIONS.lineCap,
  lineJoin: LINE_STYLE_DEFAULT_OPTIONS.lineJoin,
  lineDashOffset: LINE_STYLE_DEFAULT_OPTIONS.lineDashOffset,
  miterLimit: LINE_STYLE_DEFAULT_OPTIONS.miterLimit
};

export type PaintableOptions = {
  fill?: TPaint;
  fillPickable?: boolean;
  stroke?: TPaint;
  strokePickable?: boolean;
  lineWidth?: number;
  lineCap?: LineCap;
  lineJoin?: LineJoin;
  miterLimit?: number;
  lineDash?: number[];
  lineDashOffset?: number;
  cachedPaints?: TPaint[];
};

// Workaround type since we can't detect mixins in the type system well
export type PaintableNode = Path | Text;

const PAINTABLE_DRAWABLE_MARK_FLAGS = [ 'fill', 'stroke', 'lineWidth', 'lineOptions', 'cachedPaints' ];

const Paintable = memoize( <SuperType extends Constructor<Node>>( type: SuperType ) => {
  assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should mix Paintable' );

  return class PaintableMixin extends type {

    // (scenery-internal)
    public _fill: TPaint;
    public _fillPickable: boolean;

    // (scenery-internal)
    public _stroke: TPaint;
    public _strokePickable: boolean;

    // (scenery-internal)
    public _cachedPaints: Paint[];
    public _lineDrawingStyles: LineStyles;

    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );

      assertHasProperties( this, [ '_drawables' ] );

      this._fill = DEFAULT_OPTIONS.fill;
      this._fillPickable = DEFAULT_OPTIONS.fillPickable;

      this._stroke = DEFAULT_OPTIONS.stroke;
      this._strokePickable = DEFAULT_OPTIONS.strokePickable;

      this._cachedPaints = [];
      this._lineDrawingStyles = new LineStyles();
    }

    /**
     * Sets the fill color for the Node.
     *
     * The fill determines the appearance of the interior part of a Path or Text.
     *
     * Please use null for indicating "no fill" (that is the default). Strings and Scenery Color objects can be
     * provided for a single-color flat appearance, and can be wrapped with an Axon Property. Gradients and patterns
     * can also be provided.
     */
    public setFill( fill: TPaint ): this {
      assert && assert( PaintDef.isPaintDef( fill ), 'Invalid fill type' );

      if ( assert && typeof fill === 'string' ) {
        Color.checkPaintString( fill );
      }

      // Instance equality used here since it would be more expensive to parse all CSS
      // colors and compare every time the fill changes. Right now, usually we don't have
      // to parse CSS colors. See https://github.com/phetsims/scenery/issues/255
      if ( this._fill !== fill ) {
        this._fill = fill;

        this.invalidateFill();
      }
      return this;
    }

    public set fill( value: TPaint ) { this.setFill( value ); }

    public get fill(): TPaint { return this.getFill(); }

    /**
     * Returns the fill (if any) for this Node.
     */
    public getFill(): TPaint {
      return this._fill;
    }

    /**
     * Returns whether there is a fill applied to this Node.
     */
    public hasFill(): boolean {
      return this.getFillValue() !== null;
    }

    /**
     * Returns a property-unwrapped fill if applicable.
     */
    public getFillValue(): null | string | Color | LinearGradient | RadialGradient | Pattern {
      const fill = this.getFill();

      return fill instanceof ReadOnlyProperty ? fill.get() : fill;
    }

    public get fillValue(): null | string | Color | LinearGradient | RadialGradient | Pattern { return this.getFillValue(); }

    /**
     * Sets the stroke color for the Node.
     *
     * The stroke determines the appearance of the region along the boundary of the Path or Text. The shape of the
     * stroked area depends on the base shape (that of the Path or Text) and multiple parameters:
     * lineWidth/lineCap/lineJoin/miterLimit/lineDash/lineDashOffset. It will be drawn on top of any fill on the
     * same Node.
     *
     * Please use null for indicating "no stroke" (that is the default). Strings and Scenery Color objects can be
     * provided for a single-color flat appearance, and can be wrapped with an Axon Property. Gradients and patterns
     * can also be provided.
     */
    public setStroke( stroke: TPaint ): this {
      assert && assert( PaintDef.isPaintDef( stroke ), 'Invalid stroke type' );

      if ( assert && typeof stroke === 'string' ) {
        Color.checkPaintString( stroke );
      }

      // Instance equality used here since it would be more expensive to parse all CSS
      // colors and compare every time the fill changes. Right now, usually we don't have
      // to parse CSS colors. See https://github.com/phetsims/scenery/issues/255
      if ( this._stroke !== stroke ) {
        this._stroke = stroke;

        if ( assert && stroke instanceof Paint && stroke.transformMatrix ) {
          const scaleVector = stroke.transformMatrix.getScaleVector();
          assert( Math.abs( scaleVector.x - scaleVector.y ) < 1e-7, 'You cannot specify a pattern or gradient to a stroke that does not have a symmetric scale.' );
        }
        this.invalidateStroke();
      }
      return this;
    }

    public set stroke( value: TPaint ) { this.setStroke( value ); }

    public get stroke(): TPaint { return this.getStroke(); }

    /**
     * Returns the stroke (if any) for this Node.
     */
    public getStroke(): TPaint {
      return this._stroke;
    }

    /**
     * Returns whether there is a stroke applied to this Node.
     */
    public hasStroke(): boolean {
      return this.getStrokeValue() !== null;
    }

    /**
     * Returns whether there will appear to be a stroke for this Node. Properly handles the lineWidth:0 case.
     */
    public hasPaintableStroke(): boolean {
      // Should not be stroked if the lineWidth is 0, see https://github.com/phetsims/scenery/issues/658
      // and https://github.com/phetsims/scenery/issues/523
      return this.hasStroke() && this.getLineWidth() > 0;
    }

    /**
     * Returns a property-unwrapped stroke if applicable.
     */
    public getStrokeValue(): null | string | Color | LinearGradient | RadialGradient | Pattern {
      const stroke = this.getStroke();

      return stroke instanceof ReadOnlyProperty ? stroke.get() : stroke;
    }

    public get strokeValue(): null | string | Color | LinearGradient | RadialGradient | Pattern { return this.getStrokeValue(); }

    /**
     * Sets whether the fill is marked as pickable.
     */
    public setFillPickable( pickable: boolean ): this {
      if ( this._fillPickable !== pickable ) {
        this._fillPickable = pickable;

        // TODO: better way of indicating that only the Node under pointers could have changed, but no paint change is needed?
        this.invalidateFill();
      }
      return this;
    }

    public set fillPickable( value: boolean ) { this.setFillPickable( value ); }

    public get fillPickable(): boolean { return this.isFillPickable(); }

    /**
     * Returns whether the fill is marked as pickable.
     */
    public isFillPickable(): boolean {
      return this._fillPickable;
    }

    /**
     * Sets whether the stroke is marked as pickable.
     */
    public setStrokePickable( pickable: boolean ): this {

      if ( this._strokePickable !== pickable ) {
        this._strokePickable = pickable;

        // TODO: better way of indicating that only the Node under pointers could have changed, but no paint change is needed?
        this.invalidateStroke();
      }
      return this;
    }

    public set strokePickable( value: boolean ) { this.setStrokePickable( value ); }

    public get strokePickable(): boolean { return this.isStrokePickable(); }

    /**
     * Returns whether the stroke is marked as pickable.
     */
    public isStrokePickable(): boolean {
      return this._strokePickable;
    }

    /**
     * Sets the line width that will be applied to strokes on this Node.
     */
    public setLineWidth( lineWidth: number ): this {
      assert && assert( lineWidth >= 0, `lineWidth should be non-negative instead of ${lineWidth}` );

      if ( this.getLineWidth() !== lineWidth ) {
        this._lineDrawingStyles.lineWidth = lineWidth;
        this.invalidateStroke();

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyLineWidth();
        }
      }
      return this;
    }

    public set lineWidth( value: number ) { this.setLineWidth( value ); }

    public get lineWidth(): number { return this.getLineWidth(); }

    /**
     * Returns the line width that would be applied to strokes.
     */
    public getLineWidth(): number {
      return this._lineDrawingStyles.lineWidth;
    }

    /**
     * Sets the line cap style. There are three options:
     * - 'butt' (the default) stops the line at the end point
     * - 'round' draws a semicircular arc around the end point
     * - 'square' draws a square outline around the end point (like butt, but extended by 1/2 line width out)
     */
    public setLineCap( lineCap: LineCap ): this {
      assert && assert( lineCap === 'butt' || lineCap === 'round' || lineCap === 'square',
        `lineCap should be one of "butt", "round" or "square", not ${lineCap}` );

      if ( this._lineDrawingStyles.lineCap !== lineCap ) {
        this._lineDrawingStyles.lineCap = lineCap;
        this.invalidateStroke();

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    public set lineCap( value: LineCap ) { this.setLineCap( value ); }

    public get lineCap(): LineCap { return this.getLineCap(); }

    /**
     * Returns the line cap style (controls appearance at the start/end of paths)
     */
    public getLineCap(): LineCap {
      return this._lineDrawingStyles.lineCap;
    }

    /**
     * Sets the line join style. There are three options:
     * - 'miter' (default) joins by extending the segments out in a line until they meet. For very sharp
     *           corners, they will be chopped off and will act like 'bevel', depending on what the miterLimit is.
     * - 'round' draws a circular arc to connect the two stroked areas.
     * - 'bevel' connects with a single line segment.
     */
    public setLineJoin( lineJoin: LineJoin ): this {
      assert && assert( lineJoin === 'miter' || lineJoin === 'round' || lineJoin === 'bevel',
        `lineJoin should be one of "miter", "round" or "bevel", not ${lineJoin}` );

      if ( this._lineDrawingStyles.lineJoin !== lineJoin ) {
        this._lineDrawingStyles.lineJoin = lineJoin;
        this.invalidateStroke();

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    public set lineJoin( value: LineJoin ) { this.setLineJoin( value ); }

    public get lineJoin(): LineJoin { return this.getLineJoin(); }

    /**
     * Returns the current line join style (controls join appearance between drawn segments).
     */
    public getLineJoin(): LineJoin {
      return this._lineDrawingStyles.lineJoin;
    }

    /**
     * Sets the miterLimit value. This determines how sharp a corner with lineJoin: 'miter' will need to be before
     * it gets cut off to the 'bevel' behavior.
     */
    public setMiterLimit( miterLimit: number ): this {
      assert && assert( isFinite( miterLimit ), 'miterLimit should be a finite number' );

      if ( this._lineDrawingStyles.miterLimit !== miterLimit ) {
        this._lineDrawingStyles.miterLimit = miterLimit;
        this.invalidateStroke();

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    public set miterLimit( value: number ) { this.setMiterLimit( value ); }

    public get miterLimit(): number { return this.getMiterLimit(); }

    /**
     * Returns the miterLimit value.
     */
    public getMiterLimit(): number {
      return this._lineDrawingStyles.miterLimit;
    }

    /**
     * Sets the line dash pattern. Should be an array of numbers "on" and "off" alternating. An empty array
     * indicates no dashing.
     */
    public setLineDash( lineDash: number[] ): this {
      assert && assert( Array.isArray( lineDash ) && lineDash.every( n => typeof n === 'number' && isFinite( n ) && n >= 0 ),
        'lineDash should be an array of finite non-negative numbers' );

      if ( this._lineDrawingStyles.lineDash !== lineDash ) {
        this._lineDrawingStyles.lineDash = lineDash || [];
        this.invalidateStroke();

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    public set lineDash( value: number[] ) { this.setLineDash( value ); }

    public get lineDash(): number[] { return this.getLineDash(); }

    /**
     * Gets the line dash pattern. An empty array is the default, indicating no dashing.
     */
    public getLineDash(): number[] {
      return this._lineDrawingStyles.lineDash;
    }

    /**
     * Returns whether the stroke will be dashed.
     */
    public hasLineDash(): boolean {
      return !!this._lineDrawingStyles.lineDash.length;
    }

    /**
     * Sets the offset of the line dash pattern from the start of the stroke. Defaults to 0.
     */
    public setLineDashOffset( lineDashOffset: number ): this {
      assert && assert( isFinite( lineDashOffset ),
        `lineDashOffset should be a number, not ${lineDashOffset}` );

      if ( this._lineDrawingStyles.lineDashOffset !== lineDashOffset ) {
        this._lineDrawingStyles.lineDashOffset = lineDashOffset;
        this.invalidateStroke();

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    public set lineDashOffset( value: number ) { this.setLineDashOffset( value ); }

    public get lineDashOffset(): number { return this.getLineDashOffset(); }

    /**
     * Returns the offset of the line dash pattern from the start of the stroke.
     */
    public getLineDashOffset(): number {
      return this._lineDrawingStyles.lineDashOffset;
    }

    /**
     * Sets the LineStyles object (it determines stroke appearance). The passed-in object will be mutated as needed.
     */
    public setLineStyles( lineStyles: LineStyles ): this {
      this._lineDrawingStyles = lineStyles;
      this.invalidateStroke();
      return this;
    }

    public set lineStyles( value: LineStyles ) { this.setLineStyles( value ); }

    public get lineStyles(): LineStyles { return this.getLineStyles(); }

    /**
     * Returns the composite {LineStyles} object, that determines stroke appearance.
     */
    public getLineStyles(): LineStyles {
      return this._lineDrawingStyles;
    }

    /**
     * Sets the cached paints to the input array (a defensive copy). Note that it also filters out fills that are
     * not considered paints (e.g. strings, Colors, etc.).
     *
     * When this Node is displayed in SVG, it will force the presence of the cached paint to be stored in the SVG's
     * <defs> element, so that we can switch quickly to use the given paint (instead of having to create it on the
     * SVG-side whenever the switch is made).
     *
     * Also note that duplicate paints are acceptable, and don't need to be filtered out before-hand.
     */
    public setCachedPaints( paints: TPaint[] ): this {
      this._cachedPaints = paints.filter( ( paint: TPaint ): paint is Paint => paint instanceof Paint );

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyCachedPaints();
      }

      return this;
    }

    public set cachedPaints( value: TPaint[] ) { this.setCachedPaints( value ); }

    public get cachedPaints(): TPaint[] { return this.getCachedPaints(); }

    /**
     * Returns the cached paints.
     */
    public getCachedPaints(): TPaint[] {
      return this._cachedPaints;
    }

    /**
     * Adds a cached paint. Does nothing if paint is just a normal fill (string, Color), but for gradients and
     * patterns, it will be made faster to switch to.
     *
     * When this Node is displayed in SVG, it will force the presence of the cached paint to be stored in the SVG's
     * <defs> element, so that we can switch quickly to use the given paint (instead of having to create it on the
     * SVG-side whenever the switch is made).
     *
     * Also note that duplicate paints are acceptable, and don't need to be filtered out before-hand.
     */
    public addCachedPaint( paint: TPaint ): void {
      if ( paint instanceof Paint ) {
        this._cachedPaints.push( paint );

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyCachedPaints();
        }
      }
    }

    /**
     * Removes a cached paint. Does nothing if paint is just a normal fill (string, Color), but for gradients and
     * patterns it will remove any existing cached paint. If it was added more than once, it will need to be removed
     * more than once.
     *
     * When this Node is displayed in SVG, it will force the presence of the cached paint to be stored in the SVG's
     * <defs> element, so that we can switch quickly to use the given paint (instead of having to create it on the
     * SVG-side whenever the switch is made).
     */
    public removeCachedPaint( paint: TPaint ): void {
      if ( paint instanceof Paint ) {
        assert && assert( _.includes( this._cachedPaints, paint ) );

        arrayRemove( this._cachedPaints, paint );

        const stateLen = this._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyCachedPaints();
        }
      }
    }

    /**
     * Applies the fill to a Canvas context wrapper, before filling. (scenery-internal)
     */
    public beforeCanvasFill( wrapper: CanvasContextWrapper ): void {
      assert && assert( this.getFillValue() !== null );

      const fillValue = this.getFillValue()!;

      wrapper.setFillStyle( fillValue );
      // @ts-expect-error - For performance, we could check this by ruling out string and 'transformMatrix' in fillValue
      if ( fillValue.transformMatrix ) {
        wrapper.context.save();
        // @ts-expect-error
        fillValue.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    }

    /**
     * Un-applies the fill to a Canvas context wrapper, after filling. (scenery-internal)
     */
    public afterCanvasFill( wrapper: CanvasContextWrapper ): void {
      const fillValue = this.getFillValue();

      // @ts-expect-error
      if ( fillValue.transformMatrix ) {
        wrapper.context.restore();
      }
    }

    /**
     * Applies the stroke to a Canvas context wrapper, before stroking. (scenery-internal)
     */
    public beforeCanvasStroke( wrapper: CanvasContextWrapper ): void {
      const strokeValue = this.getStrokeValue();

      // TODO: is there a better way of not calling so many things on each stroke?
      wrapper.setStrokeStyle( this._stroke );
      wrapper.setLineCap( this.getLineCap() );
      wrapper.setLineJoin( this.getLineJoin() );

      // @ts-expect-error - for performance
      if ( strokeValue.transformMatrix ) {

        // @ts-expect-error
        const scaleVector: Vector2 = strokeValue.transformMatrix.getScaleVector();
        assert && assert( Math.abs( scaleVector.x - scaleVector.y ) < 1e-7, 'You cannot specify a pattern or gradient to a stroke that does not have a symmetric scale.' );
        const matrixMultiplier = 1 / scaleVector.x;

        wrapper.context.save();
        // @ts-expect-error
        strokeValue.transformMatrix.canvasAppendTransform( wrapper.context );

        wrapper.setLineWidth( this.getLineWidth() * matrixMultiplier );
        wrapper.setMiterLimit( this.getMiterLimit() * matrixMultiplier );
        wrapper.setLineDash( this.getLineDash().map( dash => dash * matrixMultiplier ) );
        wrapper.setLineDashOffset( this.getLineDashOffset() * matrixMultiplier );
      }
      else {
        wrapper.setLineWidth( this.getLineWidth() );
        wrapper.setMiterLimit( this.getMiterLimit() );
        wrapper.setLineDash( this.getLineDash() );
        wrapper.setLineDashOffset( this.getLineDashOffset() );
      }
    }

    /**
     * Un-applies the stroke to a Canvas context wrapper, after stroking. (scenery-internal)
     */
    public afterCanvasStroke( wrapper: CanvasContextWrapper ): void {
      const strokeValue = this.getStrokeValue();

      // @ts-expect-error - for performance
      if ( strokeValue.transformMatrix ) {
        wrapper.context.restore();
      }
    }

    /**
     * If applicable, returns the CSS color for the fill.
     */
    public getCSSFill(): string {
      const fillValue = this.getFillValue();
      // if it's a Color object, get the corresponding CSS
      // 'transparent' will make us invisible if the fill is null
      // @ts-expect-error - toCSS checks for color, left for performance
      return fillValue ? ( fillValue.toCSS ? fillValue.toCSS() : fillValue ) : 'transparent';
    }

    /**
     * If applicable, returns the CSS color for the stroke.
     */
    public getSimpleCSSStroke(): string {
      const strokeValue = this.getStrokeValue();
      // if it's a Color object, get the corresponding CSS
      // 'transparent' will make us invisible if the fill is null
      // @ts-expect-error - toCSS checks for color, left for performance
      return strokeValue ? ( strokeValue.toCSS ? strokeValue.toCSS() : strokeValue ) : 'transparent';
    }

    /**
     * Returns the fill-specific property string for use with toString(). (scenery-internal)
     *
     * @param spaces - Whitespace to add
     * @param result
     */
    public appendFillablePropString( spaces: string, result: string ): string {
      if ( this._fill ) {
        if ( result ) {
          result += ',\n';
        }
        if ( typeof this.getFillValue() === 'string' ) {
          result += `${spaces}fill: '${this.getFillValue()}'`;
        }
        else {
          result += `${spaces}fill: ${this.getFillValue()}`;
        }
      }

      return result;
    }

    /**
     * Returns the stroke-specific property string for use with toString(). (scenery-internal)
     *
     * @param spaces - Whitespace to add
     * @param result
     */
    public appendStrokablePropString( spaces: string, result: string ): string {
      function addProp( key: string, value: string, nowrap?: boolean ): void {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += `${spaces + key}: '${value}'`;
        }
        else {
          result += `${spaces + key}: ${value}`;
        }
      }

      if ( this._stroke ) {
        const defaultStyles = new LineStyles();
        const strokeValue = this.getStrokeValue();
        if ( typeof strokeValue === 'string' ) {
          addProp( 'stroke', strokeValue );
        }
        else {
          addProp( 'stroke', strokeValue ? strokeValue.toString() : 'null', true );
        }

        _.each( [ 'lineWidth', 'lineCap', 'miterLimit', 'lineJoin', 'lineDashOffset' ], prop => {
          // @ts-expect-error
          if ( this[ prop ] !== defaultStyles[ prop ] ) {
            // @ts-expect-error
            addProp( prop, this[ prop ] );
          }
        } );

        if ( this.lineDash.length ) {
          addProp( 'lineDash', JSON.stringify( this.lineDash ), true );
        }
      }

      return result;
    }

    /**
     * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
     * current fill options. (scenery-internal)
     *
     * This will be used for all types that directly mix in Paintable (i.e. Path and Text), but may be overridden
     * by subtypes.
     *
     * @returns - Renderer bitmask, see Renderer for details
     */
    public getFillRendererBitmask(): number {
      let bitmask = 0;

      // Safari 5 has buggy issues with SVG gradients
      if ( !( isSafari5 && this._fill instanceof Gradient ) ) {
        bitmask |= Renderer.bitmaskSVG;
      }

      // we always have Canvas support?
      bitmask |= Renderer.bitmaskCanvas;

      if ( !this.hasFill() ) {
        // if there is no fill, it is supported by DOM and WebGL
        bitmask |= Renderer.bitmaskDOM;
        bitmask |= Renderer.bitmaskWebGL;
      }
      else if ( this._fill instanceof Pattern ) {
        // no pattern support for DOM or WebGL (for now!)
      }
      else if ( this._fill instanceof Gradient ) {
        // no gradient support for DOM or WebGL (for now!)
      }
      else {
        // solid fills always supported for DOM and WebGL
        bitmask |= Renderer.bitmaskDOM;
        bitmask |= Renderer.bitmaskWebGL;
      }

      return bitmask;
    }

    /**
     * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
     * current stroke options. (scenery-internal)
     *
     * This will be used for all types that directly mix in Paintable (i.e. Path and Text), but may be overridden
     * by subtypes.
     *
     * @returns - Renderer bitmask, see Renderer for details
     */
    public getStrokeRendererBitmask(): number {
      let bitmask = 0;

      bitmask |= Renderer.bitmaskCanvas;

      // always have SVG support (for now?)
      bitmask |= Renderer.bitmaskSVG;

      if ( !this.hasStroke() ) {
        // allow DOM support if there is no stroke (since the fill will determine what is available)
        bitmask |= Renderer.bitmaskDOM;
        bitmask |= Renderer.bitmaskWebGL;
      }

      return bitmask;
    }

    /**
     * Invalidates our current fill, triggering recomputation of anything that depended on the old fill's value
     */
    public invalidateFill(): void {
      this.invalidateSupportedRenderers();

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyFill();
      }
    }

    /**
     * Invalidates our current stroke, triggering recomputation of anything that depended on the old stroke's value
     */
    public invalidateStroke(): void {
      this.invalidateSupportedRenderers();

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TPaintableDrawable ).markDirtyStroke();
      }
    }
  };
} );

scenery.register( 'Paintable', Paintable );

// @ts-expect-error
Paintable.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

export {
  Paintable as default,
  PAINTABLE_DRAWABLE_MARK_FLAGS,
  PAINTABLE_OPTION_KEYS,
  DEFAULT_OPTIONS,
  DEFAULT_OPTIONS as PAINTABLE_DEFAULT_OPTIONS
};
