// Copyright 2013-2022, University of Colorado Boulder

/**
 * Displays text that can be filled/stroked.
 *
 * For many font/text-related properties, it's helpful to understand the CSS equivalents (http://www.w3.org/TR/css3-fonts/).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import StringProperty from '../../../axon/js/StringProperty.js';
import TinyForwardingProperty from '../../../axon/js/TinyForwardingProperty.js';
import escapeHTML from '../../../phet-core/js/escapeHTML.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import merge from '../../../phet-core/js/merge.js';
import platform from '../../../phet-core/js/platform.js';
import Tandem from '../../../tandem/js/Tandem.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NumberIO from '../../../tandem/js/types/NumberIO.js';
import VoidIO from '../../../tandem/js/types/VoidIO.js';
import IProperty from '../../../axon/js/IProperty.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import { scenery, Font, FontStyle, FontWeight, FontStretch, ITextDrawable, CanvasContextWrapper, Instance, TextBounds, DOMSelfDrawable, SVGSelfDrawable, CanvasSelfDrawable, Renderer, TextCanvasDrawable, TextDOMDrawable, TextSVGDrawable, Node, NodeOptions, Paintable, PAINTABLE_OPTION_KEYS, PAINTABLE_DRAWABLE_MARK_FLAGS, PaintableOptions } from '../imports.js';

// constants
const TEXT_OPTION_KEYS = [
  'boundsMethod', // {string} - Sets how bounds are determined for text, see setBoundsMethod() for more documentation
  'textProperty', // {Property.<string>|null} - Sets forwarding of the textProperty, see setTextProperty() for more documentation
  'text', // {string|number} - Sets the text to be displayed, see setText() for more documentation
  'font', // {Font|string} - Sets the font used for the text, see setFont() for more documentation
  'fontWeight', // {string|number} - Sets the weight of the current font, see setFont() for more documentation
  'fontFamily', // {string} - Sets the family of the current font, see setFont() for more documentation
  'fontStretch', // {string} - Sets the stretch of the current font, see setFont() for more documentation
  'fontStyle', // {string} - Sets the style of the current font, see setFont() for more documentation
  'fontSize' // {string|number} - Sets the size of the current font, see setFont() for more documentation
];

const TEXT_PROPERTY_TANDEM_NAME = 'textProperty';

// SVG bounds seems to be malfunctioning for Safari 5. Since we don't have a reproducible test machine for
// fast iteration, we'll guess the user agent and use DOM bounds instead of SVG.
// Hopefully the two constraints rule out any future Safari versions (fairly safe, but not impossible!)
// @private {boolean}
const useDOMAsFastBounds = window.navigator.userAgent.indexOf( 'like Gecko) Version/5' ) !== -1 &&
                           window.navigator.userAgent.indexOf( 'Safari/' ) !== -1;

export type TextBoundsMethod = 'fast' | 'fastCanvas' | 'accurate' | 'hybrid';
type SelfOptions = {
  boundsMethod?: TextBoundsMethod;
  textProperty?: IProperty<string> | null;
  text?: string | number;
  font?: Font | string;
  fontWeight?: string | number;
  fontFamily?: string;
  fontStretch?: string;
  fontStyle?: string;
  fontSize?: string | number;
};
export type TextOptions = SelfOptions & PaintableOptions & NodeOptions;

export default class Text extends Paintable( Node ) {

  // The text to display
  _textProperty: TinyForwardingProperty<string>;

  // The font with which to display the text.
  _font: Font;

  _boundsMethod: TextBoundsMethod;

  // Whether the text is rendered as HTML or not. if defined (in a subtype constructor), use that value instead
  _isHTML: boolean;

  // The actual string displayed (can have non-breaking spaces and embedding marks rewritten).
  // When this is null, its value needs to be recomputed
  _cachedRenderedText: string | null;

  // (phet-io) - property name avoids namespace of the Node setter
  private textTandem: Tandem;

  /**
   * @param text - See setText() for more documentation
   * @param [options] - Text-specific options are documented in TEXT_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  constructor( text: string | number, options?: TextOptions ) {
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    super();

    // We'll initialize this by mutating.
    this._textProperty = new TinyForwardingProperty( '', true, this.onTextPropertyChange.bind( this ) );
    this._font = Font.DEFAULT;
    this._boundsMethod = 'hybrid';
    this._isHTML = false; // TODO: clean this up
    this._cachedRenderedText = null;

    const definedOptions = extendDefined( {
      fill: '#000000', // Default to black filled text
      text: text,
      tandem: Tandem.OPTIONAL,
      phetioType: Text.TextIO
    }, options );

    this.textTandem = definedOptions.tandem;

    this.mutate( definedOptions );

    this.invalidateSupportedRenderers(); // takes care of setting up supported renderers
  }

  override mutate( options?: TextOptions ) {
    // @ts-ignore
    if ( assert && options.hasOwnProperty( 'text' ) && options.hasOwnProperty( 'textProperty' ) ) {
      // @ts-ignore
      assert && assert( options.textProperty.value === options.text, 'If both text and textProperty are provided, then values should match' );
    }
    return super.mutate( options );
  }

  /**
   * Sets the text displayed by our node.
   *
   * @param text - The text to display. If it's a number, it will be cast to a string
   */
  setText( text: string | number ): this {
    assert && assert( text !== null && text !== undefined, 'Text should be defined and non-null. Use the empty string if needed.' );
    assert && assert( typeof text === 'number' || typeof text === 'string', 'text should be a string or number' );

    // cast it to a string (for numbers, etc., and do it before the change guard so we don't accidentally trigger on non-changed text)
    text = `${text}`;

    this._textProperty.set( text );

    return this;
  }

  set text( value: string | number ) { this.setText( value ); }

  /**
   * Returns the text displayed by our node.
   *
   * NOTE: If a number was provided to setText(), it will not be returned as a number here.
   *
   * @returns {string}
   */
  getText(): string {
    return this._textProperty.value;
  }

  get text(): string { return this.getText(); }

  /**
   * Returns a potentially modified version of this.text, where spaces are replaced with non-breaking spaces,
   * and embedding marks are potentially simplified.
   */
  getRenderedText(): string {
    if ( this._cachedRenderedText === null ) {
      // Using the non-breaking space (&nbsp;) encoded as 0x00A0 in UTF-8
      this._cachedRenderedText = this.text.replace( ' ', '\xA0' );

      if ( platform.edge ) {
        // Simplify embedding marks to work around an Edge bug, see https://github.com/phetsims/scenery/issues/520
        this._cachedRenderedText = Text.simplifyEmbeddingMarks( this._cachedRenderedText );
      }
    }

    return this._cachedRenderedText!;
  }

  get renderedText(): string { return this.getRenderedText(); }

  /**
   * Called when our text Property changes values.
   */
  private onTextPropertyChange() {
    this._cachedRenderedText = null;

    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      ( this._drawables[ i ] as unknown as ITextDrawable ).markDirtyText();
    }

    this.invalidateText();
  }

  /**
   * See documentation for Node.setVisibleProperty, except this is for the text string.
   */
  setTextProperty( newTarget: IProperty<string> | null ): this {
    return this._textProperty.setTargetProperty( this, TEXT_PROPERTY_TANDEM_NAME, newTarget );
  }

  set textProperty( property: IProperty<string> | null ) { this.setTextProperty( property ); }

  /**
   * Like Node.getVisibleProperty(), but for the text string. Note this is not the same as the Property provided in
   * setTextProperty. Thus is the nature of TinyForwardingProperty.
   */
  getTextProperty(): IProperty<string> {
    return this._textProperty;
  }

  get textProperty(): IProperty<string> { return this.getTextProperty(); }

  /**
   * See documentation and comments in Node.initializePhetioObject
   * @param {Object} baseOptions
   * @param {Object} config
   * @protected
   */
  override initializePhetioObject( baseOptions: any, config: TextOptions ) {

    config = merge( {
      textPropertyOptions: null
    }, config );

    // Track this, so we only override our textProperty once.
    const wasInstrumented = this.isPhetioInstrumented();

    super.initializePhetioObject( baseOptions, config );

    if ( Tandem.PHET_IO_ENABLED && !wasInstrumented && this.isPhetioInstrumented() ) {
      this._textProperty.initializePhetio( this, TEXT_PROPERTY_TANDEM_NAME, () => new StringProperty( this.text, merge( {

          // by default, use the value from the Node
          phetioReadOnly: this.phetioReadOnly,
          tandem: this.tandem.createTandem( TEXT_PROPERTY_TANDEM_NAME ),
          phetioDocumentation: 'Property for the displayed text'
        // @ts-ignore --- how to handle this?
        }, config.textPropertyOptions ) )
      );
    }
  }

  /**
   * Sets the method that is used to determine bounds from the text.
   *
   * Possible values:
   * - 'fast' - Measures using SVG, can be inaccurate. Can't be rendered in Canvas.
   * - 'fastCanvas' - Like 'fast', but allows rendering in Canvas.
   * - 'accurate' - Recursively renders to a Canvas to accurately determine bounds. Slow, but works with all renderers.
   * - 'hybrid' - [default] Cache SVG height, and uses Canvas measureText for the width.
   *
   * TODO: deprecate fast/fastCanvas options?
   *
   * NOTE: Most of these are unfortunately not hard guarantees that content is all inside of the returned bounds.
   *       'accurate' should probably be the only one where that guarantee can be assumed. Things like cyrillic in
   *       italic, combining marks and other unicode features can fail to be detected. This is particularly relevant
   *       for the height, as certain stacked accent marks or descenders can go outside of the prescribed range,
   *       and fast/canvasCanvas/hybrid will always return the same vertical bounds (top and bottom) for a given font
   *       when the text isn't the empty string.
   */
  setBoundsMethod( method: TextBoundsMethod ): this {
    assert && assert( method === 'fast' || method === 'fastCanvas' || method === 'accurate' || method === 'hybrid', 'Unknown Text boundsMethod' );
    if ( method !== this._boundsMethod ) {
      this._boundsMethod = method;
      this.invalidateSupportedRenderers();

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as ITextDrawable ).markDirtyBounds();
      }

      this.invalidateText();

      this.rendererSummaryRefreshEmitter.emit(); // whether our self bounds are valid may have changed
    }
    return this;
  }

  set boundsMethod( value: TextBoundsMethod ) { this.setBoundsMethod( value ); }

  /**
   * Returns the current method to estimate the bounds of the text. See setBoundsMethod() for more information.
   */
  getBoundsMethod(): TextBoundsMethod {
    return this._boundsMethod;
  }

  get boundsMethod(): TextBoundsMethod { return this.getBoundsMethod(); }

  /**
   * Returns a bitmask representing the supported renderers for the current configuration of the Text node.
   *
   * @returns - A bitmask that includes supported renderers, see Renderer for details.
   */
  protected getTextRendererBitmask(): number {
    let bitmask = 0;

    // canvas support (fast bounds may leak out of dirty rectangles)
    if ( this._boundsMethod !== 'fast' && !this._isHTML ) {
      bitmask |= Renderer.bitmaskCanvas;
    }
    if ( !this._isHTML ) {
      bitmask |= Renderer.bitmaskSVG;
    }

    // fill and stroke will determine whether we have DOM text support
    bitmask |= Renderer.bitmaskDOM;

    return bitmask;
  }

  /**
   * Triggers a check and update for what renderers the current configuration supports.
   * This should be called whenever something that could potentially change supported renderers happen (which can
   * be isHTML, boundsMethod, etc.)
   */
  override invalidateSupportedRenderers() {
    this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getTextRendererBitmask() );
  }

  /**
   * Notifies that something about the text's potential bounds have changed (different text, different stroke or font,
   * etc.)
   */
  private invalidateText() {
    this.invalidateSelf();

    // TODO: consider replacing this with a general dirty flag notification, and have DOM update bounds every frame?
    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      ( this._drawables[ i ] as unknown as ITextDrawable ).markDirtyBounds();
    }

    // we may have changed renderers if parameters were changed!
    this.invalidateSupportedRenderers();
  }

  /**
   * Computes a more efficient selfBounds for our Text.
   *
   * @returns - Whether the self bounds changed.
   */
  protected override updateSelfBounds(): boolean {
    // TODO: don't create another Bounds2 object just for this!
    let selfBounds;

    // investigate http://mudcu.be/journal/2011/01/html5-typographic-metrics/
    if ( this._isHTML || ( useDOMAsFastBounds && this._boundsMethod !== 'accurate' ) ) {
      selfBounds = TextBounds.approximateDOMBounds( this._font, this.getDOMTextNode() );
    }
    else if ( this._boundsMethod === 'hybrid' ) {
      selfBounds = TextBounds.approximateHybridBounds( this._font, this.renderedText );
    }
    else if ( this._boundsMethod === 'accurate' ) {
      selfBounds = TextBounds.accurateCanvasBounds( this );
    }
    else {
      assert && assert( this._boundsMethod === 'fast' || this._boundsMethod === 'fastCanvas' );
      selfBounds = TextBounds.approximateSVGBounds( this._font, this.renderedText );
    }

    // for now, just add extra on, ignoring the possibility of mitered joints passing beyond
    if ( this.hasStroke() ) {
      selfBounds.dilate( this.getLineWidth() / 2 );
    }

    const changed = !selfBounds.equals( this.selfBoundsProperty._value );
    if ( changed ) {
      this.selfBoundsProperty._value.set( selfBounds );
    }
    return changed;
  }

  /**
   * Called from (and overridden in) the Paintable trait, invalidates our current stroke, triggering recomputation of
   * anything that depended on the old stroke's value. (scenery-internal)
   */
  override invalidateStroke() {
    // stroke can change both the bounds and renderer
    this.invalidateText();

    super.invalidateStroke();
  }

  /**
   * Called from (and overridden in) the Paintable trait, invalidates our current fill, triggering recomputation of
   * anything that depended on the old fill's value. (scenery-internal)
   */
  override invalidateFill() {
    // fill type can change the renderer (gradient/fill not supported by DOM)
    this.invalidateText();

    super.invalidateFill();
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  override canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ) {
    //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
    TextCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a DOM drawable for this Text. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  override createDOMDrawable( renderer: number, instance: Instance ): DOMSelfDrawable {
    // @ts-ignore
    return TextDOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a SVG drawable for this Text. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  override createSVGDrawable( renderer: number, instance: Instance ): SVGSelfDrawable {
    // @ts-ignore
    return TextSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Text. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  override createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-ignore
    return TextCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Returns a DOM element that contains the specified text. (scenery-internal)
   *
   * This is needed since we have to handle HTML text differently.
   */
  getDOMTextNode(): any {
    if ( this._isHTML ) {
      const span = document.createElement( 'span' );
      span.innerHTML = this.text;
      return span;
    }
    else {
      return document.createTextNode( this.renderedText );
    }
  }

  /**
   * Returns a bounding box that should contain all self content in the local coordinate frame (our normal self bounds
   * aren't guaranteed this for Text)
   *
   * We need to add additional padding around the text when the text is in a container that could clip things badly
   * if the text is larger than the normal bounds computation.
   */
  override getSafeSelfBounds(): Bounds2 {
    const expansionFactor = 1; // we use a new bounding box with a new size of size * ( 1 + 2 * expansionFactor )

    const selfBounds = this.getSelfBounds();

    // NOTE: we'll keep this as an estimate for the bounds including stroke miters
    return selfBounds.dilatedXY( expansionFactor * selfBounds.width, expansionFactor * selfBounds.height );
  }

  /**
   * Sets the font of the Text node.
   *
   * This can either be a Scenery Font object, or a string. The string format is described by Font's constructor, and
   * is basically the CSS3 font shortcut format. If a string is provided, it will be wrapped with a new (immutable)
   * Scenery Font object.
   */
  setFont( font: Font | string ): this {
    assert && assert( font instanceof Font || typeof font === 'string',
      'Fonts provided to setFont should be a Font object or a string in the CSS3 font shortcut format' );

    // We need to detect whether things have updated in a different way depending on whether we are passed a string
    // or a Font object.
    const changed = font !== ( ( typeof font === 'string' ) ? this._font.toCSS() : this._font );
    if ( changed ) {
      // Wrap so that our _font is of type {Font}
      this._font = ( typeof font === 'string' ) ? Font.fromCSS( font ) : font;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as ITextDrawable ).markDirtyFont();
      }

      this.invalidateText();
    }
    return this;
  }

  set font( value: Font | string ) { this.setFont( value ); }

  /**
   * Returns a string representation of the current Font.
   *
   * This returns the CSS3 font shortcut that is a possible input to setFont(). See Font's constructor for detailed
   * information on the ordering of information.
   *
   * NOTE: If a Font object was provided to setFont(), this will not currently return it.
   * TODO: Can we refactor so we can have access to (a) the Font object, and possibly (b) the initially provided value.
   */
  getFont(): string {
    return this._font.getFont();
  }

  get font(): string { return this.getFont(); }

  /**
   * Sets the weight of this node's font.
   *
   * The font weight supports the following options:
   *   'normal', 'bold', 'bolder', 'lighter', '100', '200', '300', '400', '500', '600', '700', '800', '900',
   *   or a number that when cast to a string will be one of the strings above.
   */
  setFontWeight( weight: FontWeight | number ): this {
    return this.setFont( this._font.copy( {
      weight: weight
    } ) );
  }

  set fontWeight( value: FontWeight | number ) { this.setFontWeight( value ); }

  /**
   * Returns the weight of this node's font, see setFontWeight() for details.
   *
   * NOTE: If a numeric weight was passed in, it has been cast to a string, and a string will be returned here.
   */
  getFontWeight(): FontWeight {
    return this._font.getWeight();
  }

  get fontWeight(): FontWeight { return this.getFontWeight(); }

  /**
   * Sets the family of this node's font.
   *
   * @param family - A comma-separated list of families, which can include generic families (preferably at
   *                 the end) such as 'serif', 'sans-serif', 'cursive', 'fantasy' and 'monospace'. If there
   *                 is any question about escaping (such as spaces in a font name), the family should be
   *                 surrounded by double quotes.
   */
  setFontFamily( family: string ): this {
    return this.setFont( this._font.copy( {
      family: family
    } ) );
  }

  set fontFamily( value: string ) { this.setFontFamily( value ); }

  /**
   * Returns the family of this node's font, see setFontFamily() for details.
   */
  getFontFamily(): string {
    return this._font.getFamily();
  }

  get fontFamily(): string { return this.getFontFamily(); }

  /**
   * Sets the stretch of this node's font.
   *
   * The font stretch supports the following options:
   *   'normal', 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed',
   *   'semi-expanded', 'expanded', 'extra-expanded' or 'ultra-expanded'
   */
  setFontStretch( stretch: FontStretch ): this {
    return this.setFont( this._font.copy( {
      stretch: stretch
    } ) );
  }

  set fontStretch( value: FontStretch ) { this.setFontStretch( value ); }

  /**
   * Returns the stretch of this node's font, see setFontStretch() for details.
   */
  getFontStretch(): FontStretch {
    return this._font.getStretch();
  }

  get fontStretch(): FontStretch { return this.getFontStretch(); }

  /**
   * Sets the style of this node's font.
   *
   * The font style supports the following options: 'normal', 'italic' or 'oblique'
   */
  setFontStyle( style: FontStyle ): this {
    return this.setFont( this._font.copy( {
      style: style
    } ) );
  }

  set fontStyle( value: FontStyle ) { this.setFontStyle( value ); }

  /**
   * Returns the style of this node's font, see setFontStyle() for details.
   */
  getFontStyle(): FontStyle {
    return this._font.getStyle();
  }

  get fontStyle(): FontStyle { return this.getFontStyle(); }

  /**
   * Sets the size of this node's font.
   *
   * The size can either be a number (created as a quantity of 'px'), or any general CSS font-size string (for
   * example, '30pt', '5em', etc.)
   */
  setFontSize( size: string | number ): this {
    return this.setFont( this._font.copy( {
      size: size
    } ) );
  }

  set fontSize( value: string | number ) { this.setFontSize( value ); }

  /**
   * Returns the size of this node's font, see setFontSize() for details.
   *
   * NOTE: If a numeric size was passed in, it has been converted to a string with 'px', and a string will be
   * returned here.
   */
  getFontSize(): string {
    return this._font.getSize();
  }

  get fontSize(): string { return this.getFontSize(); }

  /**
   * Whether this Node itself is painted (displays something itself).
   */
  override isPainted(): boolean {
    // Always true for Text nodes
    return true;
  }

  /**
   * Whether this Node's selfBounds are considered to be valid (always containing the displayed self content
   * of this node). Meant to be overridden in subtypes when this can change (e.g. Text).
   *
   * If this value would potentially change, please trigger the event 'selfBoundsValid'.
   */
  override areSelfBoundsValid(): boolean {
    return this._boundsMethod === 'accurate';
  }

  /**
   * Override for extra information in the debugging output (from Display.getDebugHTML()). (scenery-internal)
   */
  override getDebugHTMLExtras(): string {
    return ` "${escapeHTML( this.renderedText )}"${this._isHTML ? ' (html)' : ''}`;
  }

  override dispose() {
    super.dispose();

    this._textProperty.dispose();
  }

  /**
   * Replaces embedding mark characters with visible strings. Useful for debugging for strings with embedding marks.
   *
   * @returns - With embedding marks replaced.
   */
  static embeddedDebugString( string: string ): string {
    return string.replace( /\u202a/g, '[LTR]' ).replace( /\u202b/g, '[RTL]' ).replace( /\u202c/g, '[POP]' );
  }

  /**
   * Returns a (potentially) modified string where embedding marks have been simplified.
   *
   * This simplification wouldn't usually be necessary, but we need to prevent cases like
   * https://github.com/phetsims/scenery/issues/520 where Edge decides to turn [POP][LTR] (after another [LTR]) into
   * a 'box' character, when nothing should be rendered.
   *
   * This will remove redundant nesting:
   *   e.g. [LTR][LTR]boo[POP][POP] => [LTR]boo[POP])
   * and will also combine adjacent directions:
   *   e.g. [LTR]Mail[POP][LTR]Man[POP] => [LTR]MailMan[Pop]
   *
   * Note that it will NOT combine in this way if there was a space between the two LTRs:
   *   e.g. [LTR]Mail[POP] [LTR]Man[Pop])
   * as in the general case, we'll want to preserve the break there between embeddings.
   *
   * TODO: A stack-based implementation that doesn't create a bunch of objects/closures would be nice for performance.
   */
  static simplifyEmbeddingMarks( string: string ): string {
    // First, we'll convert the string into a tree form, where each node is either a string object OR an object of the
    // node type { dir: {LTR||RTL}, children: {Array.<node>}, parent: {null|node} }. Thus each LTR...POP and RTL...POP
    // become a node with their interiors becoming children.

    type EmbedNode = {
      dir: null | '\u202a' | '\u202b';
      children: ( EmbedNode | string )[];
      parent: EmbedNode | null;
    };

    // Root node (no direction, so we preserve root LTR/RTLs)
    const root = <EmbedNode>{
      dir: null,
      children: [],
      parent: null
    };
    let current: EmbedNode = root;
    for ( let i = 0; i < string.length; i++ ) {
      const chr = string.charAt( i );

      // Push a direction
      if ( chr === LTR || chr === RTL ) {
        const node = <EmbedNode>{
          dir: chr,
          children: [],
          parent: current
        };
        current.children.push( node );
        current = node;
      }
      // Pop a direction
      else if ( chr === POP ) {
        assert && assert( current.parent, `Bad nesting of embedding marks: ${Text.embeddedDebugString( string )}` );
        current = current.parent!;
      }
      // Append characters to the current direction
      else {
        current.children.push( chr );
      }
    }
    assert && assert( current === root, `Bad nesting of embedding marks: ${Text.embeddedDebugString( string )}` );

    // Remove redundant nesting (e.g. [LTR][LTR]...[POP][POP])
    function collapseNesting( node: EmbedNode ) {
      for ( let i = node.children.length - 1; i >= 0; i-- ) {
        const child = node.children[ i ];
        if ( typeof child !== 'string' && node.dir === child.dir ) {
          node.children.splice( i, 1, ...child.children );
        }
      }
    }

    // Remove overridden nesting (e.g. [LTR][RTL]...[POP][POP]), since the outer one is not needed
    function collapseUnnecessary( node: EmbedNode ) {
      if ( node.children.length === 1 && typeof node.children[ 0 ] !== 'string' && node.children[ 0 ].dir ) {
        node.dir = node.children[ 0 ].dir;
        node.children = node.children[ 0 ].children;
      }
    }

    // Collapse adjacent matching dirs, e.g. [LTR]...[POP][LTR]...[POP]
    function collapseAdjacent( node: EmbedNode ) {
      for ( let i = node.children.length - 1; i >= 1; i-- ) {
        const previousChild = node.children[ i - 1 ];
        const child = node.children[ i ];
        if ( typeof child !== 'string' && typeof previousChild !== 'string' && child.dir && previousChild.dir === child.dir ) {
          previousChild.children = previousChild.children.concat( child.children );
          node.children.splice( i, 1 );

          // Now try to collapse adjacent items in the child, since we combined children arrays
          collapseAdjacent( previousChild );
        }
      }
    }

    // Simplifies the tree using the above functions
    function simplify( node: EmbedNode | string ) {
      if ( typeof node !== 'string' ) {
        for ( let i = 0; i < node.children.length; i++ ) {
          simplify( node.children[ i ] );
        }

        collapseUnnecessary( node );
        collapseNesting( node );
        collapseAdjacent( node );
      }

      return node;
    }

    // Turns a tree into a string
    function stringify( node: EmbedNode | string ): string {
      if ( typeof node === 'string' ) {
        return node;
      }
      const childString = node.children.map( stringify ).join( '' );
      if ( node.dir ) {
        return `${node.dir + childString}\u202c`;
      }
      else {
        return childString;
      }
    }

    return stringify( simplify( root ) );
  }

  static TextIO: IOType;
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Text.prototype._mutatorKeys = [ ...PAINTABLE_OPTION_KEYS, ...TEXT_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * (scenery-internal)
 * @override
 */
Text.prototype.drawableMarkFlags = [ ...Node.prototype.drawableMarkFlags, ...PAINTABLE_DRAWABLE_MARK_FLAGS, 'text', 'font', 'bounds' ];

scenery.register( 'Text', Text );

// Unicode embedding marks that we can combine to work around the Edge issue.
// See https://github.com/phetsims/scenery/issues/520
const LTR = '\u202a';
const RTL = '\u202b';
const POP = '\u202c';

// Initialize computation of hybrid text
TextBounds.initializeTextBounds();

Text.TextIO = new IOType( 'TextIO', {
  valueType: Text,
  supertype: Node.NodeIO,
  documentation: 'Text that is displayed in the simulation. TextIO has a nested PropertyIO.&lt;String&gt; for ' +
                 'the current string value.',
  methods: {
    setFontOptions: {
      returnType: VoidIO,
      parameterTypes: [ Font.FontIO ],
      implementation: function( this: Text, font: Font ) {
        this.setFont( font );
      },
      documentation: 'Sets font options for this TextIO instance, e.g. {size: 16, weight: bold}. If increasing the font ' +
                     'size does not make the text size larger, you may need to increase the maxWidth of the TextIO also.',
      invocableForReadOnlyElements: false
    },

    getFontOptions: {
      returnType: Font.FontIO,
      parameterTypes: [],
      implementation: function( this: Text ) {
        return this.getFont();
      },
      documentation: 'Gets font options for this TextIO instance as an object'
    },

    setMaxWidth: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( this: Text, maxWidth: number ) {
        this.setMaxWidth( maxWidth );
      },
      documentation: 'Sets the maximum width of text box. ' +
                     'If the text width exceeds maxWidth, it is scaled down to fit.',
      invocableForReadOnlyElements: false
    },

    getMaxWidth: {
      returnType: NumberIO,
      parameterTypes: [],
      implementation: function( this: Text ) {
        return this.maxWidth;
      },
      documentation: 'Gets the maximum width of text box'
    }
  }
} );
