// Copyright 2021-2022, University of Colorado Boulder

/**
 * Trait for Nodes that support a standard fill and/or stroke (e.g. Text, Path and Path subtypes).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import { LineStyles, LINE_STYLE_DEFAULT_OPTIONS, LineCap, LineJoin } from '../../../kite/js/imports.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import assertHasProperties from '../../../phet-core/js/assertHasProperties.js';
import inheritance from '../../../phet-core/js/inheritance.js';
import platform from '../../../phet-core/js/platform.js';
import memoize from '../../../phet-core/js/memoize.js';
import { scenery, Renderer, Color, PaintDef, Node, IPaint, Paint, LinearGradient, Pattern, RadialGradient, CanvasContextWrapper, Gradient, IPaintableDrawable, Path, Text } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

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

type PaintableOptions = {
  fill?: IPaint,
  fillPickable?: boolean,
  stroke?: IPaint,
  strokePickable?: boolean,
  lineWidth?: number,
  lineCap?: LineCap,
  lineJoin?: LineJoin,
  miterLimit?: number,
  lineDash?: number[],
  lineDashOffset?: number,
  cachedPaints?: IPaint[]
};

// Workaround type since we can't detect mixins in the type system well
export type PaintableNode = Path | Text;

const PAINTABLE_DRAWABLE_MARK_FLAGS = [ 'fill', 'stroke', 'lineWidth', 'lineOptions', 'cachedPaints' ];

const Paintable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should mix Paintable' );

  return class extends type {

    _fill: IPaint;
    _fillPickable: boolean;

    _stroke: IPaint;
    _strokePickable: boolean;

    _cachedPaints: Paint[];
    _lineDrawingStyles: LineStyles;

    constructor( ...args: any[] ) {
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
    setFill( fill: IPaint ): this {
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

    set fill( value: IPaint ) { this.setFill( value ); }

    /**
     * Returns the fill (if any) for this Node.
     */
    getFill(): IPaint {
      return this._fill;
    }

    get fill(): IPaint { return this.getFill(); }

    /**
     * Returns whether there is a fill applied to this Node.
     */
    hasFill(): boolean {
      return this._fill !== null;
    }

    /**
     * Returns a property-unwrapped fill if applicable.
     */
    getFillValue(): null | string | Color | LinearGradient | RadialGradient | Pattern {
      const fill = this.getFill();

      return fill instanceof Property ? fill.get() : fill;
    }

    get fillValue(): null | string | Color | LinearGradient | RadialGradient | Pattern { return this.getFillValue(); }

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
    setStroke( stroke: IPaint ): this {
      assert && assert( PaintDef.isPaintDef( stroke ), 'Invalid stroke type' );

      if ( assert && typeof stroke === 'string' ) {
        Color.checkPaintString( stroke );
      }

      // Instance equality used here since it would be more expensive to parse all CSS
      // colors and compare every time the fill changes. Right now, usually we don't have
      // to parse CSS colors. See https://github.com/phetsims/scenery/issues/255
      if ( this._stroke !== stroke ) {
        this._stroke = stroke;

        this.invalidateStroke();
      }
      return this;
    }

    set stroke( value: IPaint ) { this.setStroke( value ); }

    /**
     * Returns the stroke (if any) for this Node.
     */
    getStroke(): IPaint {
      return this._stroke;
    }

    get stroke(): IPaint { return this.getStroke(); }

    /**
     * Returns whether there is a stroke applied to this Node.
     */
    hasStroke(): boolean {
      return this._stroke !== null;
    }

    /**
     * Returns whether there will appear to be a stroke for this Node. Properly handles the lineWidth:0 case.
     */
    hasPaintableStroke(): boolean {
      // Should not be stroked if the lineWidth is 0, see https://github.com/phetsims/scenery/issues/658
      // and https://github.com/phetsims/scenery/issues/523
      return this.hasStroke() && this.getLineWidth() > 0;
    }

    /**
     * Returns a property-unwrapped stroke if applicable.
     */
    getStrokeValue(): null | string | Color | LinearGradient | RadialGradient | Pattern {
      const stroke = this.getStroke();

      return stroke instanceof Property ? stroke.get() : stroke;
    }

    get strokeValue(): null | string | Color | LinearGradient | RadialGradient | Pattern { return this.getStrokeValue(); }

    /**
     * Sets whether the fill is marked as pickable.
     */
    setFillPickable( pickable: boolean ): this {
      assert && assert( typeof pickable === 'boolean' );

      if ( this._fillPickable !== pickable ) {
        this._fillPickable = pickable;

        // TODO: better way of indicating that only the Node under pointers could have changed, but no paint change is needed?
        this.invalidateFill();
      }
      return this;
    }

    set fillPickable( value: boolean ) { this.setFillPickable( value ); }

    /**
     * Returns whether the fill is marked as pickable.
     */
    isFillPickable(): boolean {
      return this._fillPickable;
    }

    get fillPickable(): boolean { return this.isFillPickable(); }

    /**
     * Sets whether the stroke is marked as pickable.
     */
    setStrokePickable( pickable: boolean ): this {
      assert && assert( typeof pickable === 'boolean', `strokePickable should be a boolean, not ${pickable}` );

      if ( this._strokePickable !== pickable ) {
        this._strokePickable = pickable;

        // TODO: better way of indicating that only the Node under pointers could have changed, but no paint change is needed?
        this.invalidateStroke();
      }
      return this;
    }

    set strokePickable( value: boolean ) { this.setStrokePickable( value ); }

    /**
     * Returns whether the stroke is marked as pickable.
     */
    isStrokePickable(): boolean {
      return this._strokePickable;
    }

    get strokePickable(): boolean { return this.isStrokePickable(); }

    /**
     * Sets the line width that will be applied to strokes on this Node.
     */
    setLineWidth( lineWidth: number ): this {
      assert && assert( typeof lineWidth === 'number', `lineWidth should be a number, not ${lineWidth}` );
      assert && assert( lineWidth >= 0, `lineWidth should be non-negative instead of ${lineWidth}` );

      if ( this.getLineWidth() !== lineWidth ) {
        this._lineDrawingStyles.lineWidth = lineWidth;
        this.invalidateStroke();

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyLineWidth();
        }
      }
      return this;
    }

    set lineWidth( value: number ) { this.setLineWidth( value ); }

    /**
     * Returns the line width that would be applied to strokes.
     */
    getLineWidth(): number {
      return this._lineDrawingStyles.lineWidth;
    }

    get lineWidth(): number { return this.getLineWidth(); }

    /**
     * Sets the line cap style. There are three options:
     * - 'butt' (the default) stops the line at the end point
     * - 'round' draws a semicircular arc around the end point
     * - 'square' draws a square outline around the end point (like butt, but extended by 1/2 line width out)
     */
    setLineCap( lineCap: LineCap ): this {
      assert && assert( lineCap === 'butt' || lineCap === 'round' || lineCap === 'square',
        `lineCap should be one of "butt", "round" or "square", not ${lineCap}` );

      if ( this._lineDrawingStyles.lineCap !== lineCap ) {
        this._lineDrawingStyles.lineCap = lineCap;
        this.invalidateStroke();

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    set lineCap( value: LineCap ) { this.setLineCap( value ); }

    /**
     * Returns the line cap style (controls appearance at the start/end of paths)
     */
    getLineCap(): LineCap {
      return this._lineDrawingStyles.lineCap;
    }

    get lineCap(): LineCap { return this.getLineCap(); }

    /**
     * Sets the line join style. There are three options:
     * - 'miter' (default) joins by extending the segments out in a line until they meet. For very sharp
     *           corners, they will be chopped off and will act like 'bevel', depending on what the miterLimit is.
     * - 'round' draws a circular arc to connect the two stroked areas.
     * - 'bevel' connects with a single line segment.
     */
    setLineJoin( lineJoin: LineJoin ): this {
      assert && assert( lineJoin === 'miter' || lineJoin === 'round' || lineJoin === 'bevel',
        `lineJoin should be one of "miter", "round" or "bevel", not ${lineJoin}` );

      if ( this._lineDrawingStyles.lineJoin !== lineJoin ) {
        this._lineDrawingStyles.lineJoin = lineJoin;
        this.invalidateStroke();

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    set lineJoin( value: LineJoin ) { this.setLineJoin( value ); }

    /**
     * Returns the current line join style (controls join appearance between drawn segments).
     */
    getLineJoin(): LineJoin {
      return this._lineDrawingStyles.lineJoin;
    }

    get lineJoin(): LineJoin { return this.getLineJoin(); }

    /**
     * Sets the miterLimit value. This determines how sharp a corner with lineJoin: 'miter' will need to be before
     * it gets cut off to the 'bevel' behavior.
     */
    setMiterLimit( miterLimit: number ): this {
      assert && assert( typeof miterLimit === 'number' && isFinite( miterLimit ), 'miterLimit should be a finite number' );

      if ( this._lineDrawingStyles.miterLimit !== miterLimit ) {
        this._lineDrawingStyles.miterLimit = miterLimit;
        this.invalidateStroke();

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    set miterLimit( value: number ) { this.setMiterLimit( value ); }

    /**
     * Returns the miterLimit value.
     */
    getMiterLimit(): number {
      return this._lineDrawingStyles.miterLimit;
    }

    get miterLimit(): number { return this.getMiterLimit(); }

    /**
     * Sets the line dash pattern. Should be an array of numbers "on" and "off" alternating. An empty array
     * indicates no dashing.
     */
    setLineDash( lineDash: number[] ): this {
      assert && assert( Array.isArray( lineDash ) && lineDash.every( n => typeof n === 'number' && isFinite( n ) && n >= 0 ),
        'lineDash should be an array of finite non-negative numbers' );

      if ( this._lineDrawingStyles.lineDash !== lineDash ) {
        this._lineDrawingStyles.lineDash = lineDash || [];
        this.invalidateStroke();

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    set lineDash( value: number[] ) { this.setLineDash( value ); }

    /**
     * Gets the line dash pattern. An empty array is the default, indicating no dashing.
     */
    getLineDash(): number[] {
      return this._lineDrawingStyles.lineDash;
    }

    get lineDash(): number[] { return this.getLineDash(); }

    /**
     * Returns whether the stroke will be dashed.
     */
    hasLineDash(): boolean {
      return !!this._lineDrawingStyles.lineDash.length;
    }

    /**
     * Sets the offset of the line dash pattern from the start of the stroke. Defaults to 0.
     */
    setLineDashOffset( lineDashOffset: number ): this {
      assert && assert( typeof lineDashOffset === 'number' && isFinite( lineDashOffset ),
        `lineDashOffset should be a number, not ${lineDashOffset}` );

      if ( this._lineDrawingStyles.lineDashOffset !== lineDashOffset ) {
        this._lineDrawingStyles.lineDashOffset = lineDashOffset;
        this.invalidateStroke();

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyLineOptions();
        }
      }
      return this;
    }

    set lineDashOffset( value: number ) { this.setLineDashOffset( value ); }

    /**
     * Returns the offset of the line dash pattern from the start of the stroke.
     */
    getLineDashOffset(): number {
      return this._lineDrawingStyles.lineDashOffset;
    }

    get lineDashOffset(): number { return this.getLineDashOffset(); }

    /**
     * Sets the LineStyles object (it determines stroke appearance). The passed-in object will be mutated as needed.
     */
    setLineStyles( lineStyles: LineStyles ): this {
      assert && assert( lineStyles instanceof LineStyles );

      this._lineDrawingStyles = lineStyles;
      this.invalidateStroke();
      return this;
    }

    set lineStyles( value: LineStyles ) { this.setLineStyles( value ); }

    /**
     * Returns the composite {LineStyles} object, that determines stroke appearance.
     */
    getLineStyles(): LineStyles {
      return this._lineDrawingStyles;
    }

    get lineStyles(): LineStyles { return this.getLineStyles(); }

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
    setCachedPaints( paints: IPaint[] ): this {
      this._cachedPaints = paints.filter( ( paint: IPaint ): paint is Paint => paint instanceof Paint );

      const stateLen = ( this as unknown as Node )._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyCachedPaints();
      }

      return this;
    }

    set cachedPaints( value: IPaint[] ) { this.setCachedPaints( value ); }

    /**
     * Returns the cached paints.
     */
    getCachedPaints(): IPaint[] {
      return this._cachedPaints;
    }

    get cachedPaints(): IPaint[] { return this.getCachedPaints(); }

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
    addCachedPaint( paint: IPaint ) {
      if ( paint instanceof Paint ) {
        this._cachedPaints.push( paint );

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyCachedPaints();
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
    removeCachedPaint( paint: IPaint ) {
      if ( paint instanceof Paint ) {
        assert && assert( _.includes( this._cachedPaints, paint ) );

        arrayRemove( this._cachedPaints, paint );

        const stateLen = ( this as unknown as Node )._drawables.length;
        for ( let i = 0; i < stateLen; i++ ) {
          ( ( this as unknown as Node )._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyCachedPaints();
        }
      }
    }

    /**
     * Applies the fill to a Canvas context wrapper, before filling. (scenery-internal)
     */
    beforeCanvasFill( wrapper: CanvasContextWrapper ) {
      assert && assert( this.getFillValue() !== null );

      const fillValue = this.getFillValue()!;

      wrapper.setFillStyle( fillValue );
      // @ts-ignore - For performance, we could check this by ruling out string and 'transformMatrix' in fillValue
      if ( fillValue.transformMatrix ) {
        wrapper.context.save();
        // @ts-ignore
        fillValue.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    }

    /**
     * Un-applies the fill to a Canvas context wrapper, after filling. (scenery-internal)
     */
    afterCanvasFill( wrapper: CanvasContextWrapper ) {
      const fillValue = this.getFillValue();

      // @ts-ignore
      if ( fillValue.transformMatrix ) {
        wrapper.context.restore();
      }
    }

    /**
     * Applies the stroke to a Canvas context wrapper, before stroking. (scenery-internal)
     */
    beforeCanvasStroke( wrapper: CanvasContextWrapper ) {
      const strokeValue = this.getStrokeValue();

      // TODO: is there a better way of not calling so many things on each stroke?
      wrapper.setStrokeStyle( this._stroke );
      wrapper.setLineWidth( this.getLineWidth() );
      wrapper.setLineCap( this.getLineCap() );
      wrapper.setLineJoin( this.getLineJoin() );
      wrapper.setMiterLimit( this.getMiterLimit() );
      wrapper.setLineDash( this.getLineDash() );
      wrapper.setLineDashOffset( this.getLineDashOffset() );

      // @ts-ignore - for performance
      if ( strokeValue.transformMatrix ) {
        wrapper.context.save();
        // @ts-ignore
        strokeValue.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    }

    /**
     * Un-applies the stroke to a Canvas context wrapper, after stroking. (scenery-internal)
     */
    afterCanvasStroke( wrapper: CanvasContextWrapper ) {
      const strokeValue = this.getStrokeValue();

      // @ts-ignore - for performance
      if ( strokeValue.transformMatrix ) {
        wrapper.context.restore();
      }
    }

    /**
     * If applicable, returns the CSS color for the fill.
     */
    getCSSFill(): string {
      const fillValue = this.getFillValue();
      // if it's a Color object, get the corresponding CSS
      // 'transparent' will make us invisible if the fill is null
      // @ts-ignore - toCSS checks for color, left for performance
      return fillValue ? ( fillValue.toCSS ? fillValue.toCSS() : fillValue ) : 'transparent';
    }

    /**
     * If applicable, returns the CSS color for the stroke.
     */
    getSimpleCSSStroke(): string {
      const strokeValue = this.getStrokeValue();
      // if it's a Color object, get the corresponding CSS
      // 'transparent' will make us invisible if the fill is null
      // @ts-ignore - toCSS checks for color, left for performance
      return strokeValue ? ( strokeValue.toCSS ? strokeValue.toCSS() : strokeValue ) : 'transparent';
    }

    /**
     * Returns the fill-specific property string for use with toString(). (scenery-internal)
     *
     * @param spaces - Whitespace to add
     * @param result
     */
    appendFillablePropString( spaces: string, result: string ): string {
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
    appendStrokablePropString( spaces: string, result: string ): string {
      function addProp( key: string, value: any, nowrap?: boolean ) {
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
          // @ts-ignore
          if ( this[ prop ] !== defaultStyles[ prop ] ) {
            // @ts-ignore
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
    getFillRendererBitmask(): number {
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
    getStrokeRendererBitmask(): number {
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
    invalidateFill() {
      const thisNode = this as unknown as Node;

      thisNode.invalidateSupportedRenderers();

      const stateLen = thisNode._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( thisNode._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyFill();
      }
    }

    /**
     * Invalidates our current stroke, triggering recomputation of anything that depended on the old stroke's value
     */
    invalidateStroke() {
      const thisNode = this as unknown as Node;

      thisNode.invalidateSupportedRenderers();

      const stateLen = thisNode._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( thisNode._drawables[ i ] as unknown as IPaintableDrawable ).markDirtyStroke();
      }
    }
  };
} );

scenery.register( 'Paintable', Paintable );

// @ts-ignore
Paintable.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

export {
  Paintable as default,
  PAINTABLE_DRAWABLE_MARK_FLAGS,
  PAINTABLE_OPTION_KEYS,
  DEFAULT_OPTIONS,
  DEFAULT_OPTIONS as PAINTABLE_DEFAULT_OPTIONS
};
export type { PaintableOptions };
