// Copyright 2017-2022, University of Colorado Boulder

/**
 * Displays rich text by interpreting the input text as HTML, supporting a limited set of tags that prevent any
 * security vulnerabilities. It does this by parsing the input HTML and splitting it into multiple Text children
 * recursively.
 *
 * NOTE: Encoding HTML entities is required, and malformed HTML is not accepted.
 *
 * NOTE: Currently it can line-wrap at the start and end of tags. This will probably be fixed in the future to only
 *       potentially break on whitespace.
 *
 * It supports the following markup and features in the string content (in addition to other options as listed in
 * RICH_TEXT_OPTION_KEYS):
 * - <a href="{{placeholder}}"> for links (pass in { links: { placeholder: ACTUAL_HREF } })
 * - <b> and <strong> for bold text
 * - <i> and <em> for italic text
 * - <sub> and <sup> for subscripts / superscripts
 * - <u> for underlined text
 * - <s> for strikethrough text
 * - <span> tags with a dir="ltr" / dir="rtl" attribute
 * - <br> for explicit line breaks
 * - Unicode bidirectional marks (present in PhET strings) for full RTL support
 * - CSS style="..." attributes, with color and font settings, see https://github.com/phetsims/scenery/issues/807
 *
 * Examples from the scenery-phet demo:
 *
 * new RichText( 'RichText can have <b>bold</b> and <i>italic</i> text.' ),
 * new RichText( 'Can do H<sub>2</sub>O (A<sub>sub</sub> and A<sup>sup</sup>), or nesting: x<sup>2<sup>2</sup></sup>' ),
 * new RichText( 'Additionally: <span style="color: blue;">color</span>, <span style="font-size: 30px;">sizes</span>, <span style="font-family: serif;">faces</span>, <s>strikethrough</s>, and <u>underline</u>' ),
 * new RichText( 'These <b><em>can</em> <u><span style="color: red;">be</span> mixed<sup>1</sup></u></b>.' ),
 * new RichText( '\u202aHandles bidirectional text: \u202b<span style="color: #0a0;">مقابض</span> النص ثنائي <b>الاتجاه</b><sub>2</sub>\u202c\u202c' ),
 * new RichText( '\u202b\u062a\u0633\u062a (\u0632\u0628\u0627\u0646)\u202c' ),
 * new RichText( 'HTML entities need to be escaped, like &amp; and &lt;.' ),
 * new RichText( 'Supports <a href="{{phetWebsite}}"><em>links</em> with <b>markup</b></a>, and <a href="{{callback}}">links that call functions</a>.', {
 *   links: {
 *     phetWebsite: 'https://phet.colorado.edu',
 *     callback: function() {
 *       console.log( 'Link was clicked' );
 *     }
 *   }
 * } ),
 * new RichText( 'Or also <a href="https://phet.colorado.edu">links directly in the string</a>.', {
 *   links: true
 * } ),
 * new RichText( 'Links not found <a href="{{bogus}}">are ignored</a> for security.' ),
 * new HBox( {
 *   spacing: 30,
 *   children: [
 *     new RichText( 'Multi-line text with the<br>separator &lt;br&gt; and <a href="https://phet.colorado.edu">handles<br>links</a> and other <b>tags<br>across lines</b>', {
 *       links: true
 *     } ),
 *     new RichText( 'Supposedly RichText supports line wrapping. Here is a lineWrap of 300, which should probably wrap multiple times here', { lineWrap: 300 } )
 *   ]
 * } )
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TProperty from '../../../axon/js/TProperty.js';
import { PropertyOptions } from '../../../axon/js/Property.js';
import StringProperty from '../../../axon/js/StringProperty.js';
import TinyForwardingProperty from '../../../axon/js/TinyForwardingProperty.js';
import Range from '../../../dot/js/Range.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import inheritance from '../../../phet-core/js/inheritance.js';
import memoize from '../../../phet-core/js/memoize.js';
import Tandem from '../../../tandem/js/Tandem.js';
import IOType from '../../../tandem/js/types/IOType.js';
import { Color, FireListener, Font, TInputListener, TPaint, Line, Node, NodeOptions, scenery, Text, TextBoundsMethod, Voicing, VStrut, openPopup, getLineBreakRanges } from '../imports.js';
import Pool from '../../../phet-core/js/Pool.js';
import optionize, { combineOptions, EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';

// Options that can be used in the constructor, with mutate(), or directly as setters/getters
// each of these options has an associated setter, see setter methods for more documentation
const RICH_TEXT_OPTION_KEYS = [
  'boundsMethod',
  'font',
  'fill',
  'stroke',
  'lineWidth',
  'subScale',
  'subXSpacing',
  'subYOffset',
  'supScale',
  'supXSpacing',
  'supYOffset',
  'capHeightScale',
  'underlineLineWidth',
  'underlineHeightScale',
  'strikethroughLineWidth',
  'strikethroughHeightScale',
  'linkFill',
  'linkEventsHandled',
  'links',
  'align',
  'leading',
  'lineWrap',
  'textProperty',
  'text'
];

export type RichTextAlign = 'left' | 'center' | 'right';
export type RichTextHref = ( () => void ) | string;
type RichTextLinksObject = Record<string, string>;
export type RichTextLinks = RichTextLinksObject | boolean;

type SelfOptions = {
  // Sets how bounds are determined for text
  boundsMethod?: TextBoundsMethod;

  // Sets the font for the text
  font?: Font | string;

  // Sets the fill of the text
  fill?: TPaint;

  // Sets the stroke around the text
  stroke?: TPaint;

  // Sets the lineWidth around the text
  lineWidth?: number;

  // Sets the scale of any subscript elements
  subScale?: number;

  // Sets horizontal spacing before any subscript elements
  subXSpacing?: number;

  // Sets vertical offset for any subscript elements
  subYOffset?: number;

  // Sets the scale for any superscript elements
  supScale?: number;

  // Sets the horizontal offset before any superscript elements
  supXSpacing?: number;

  // Sets the vertical offset for any superscript elements
  supYOffset?: number;

  // Sets the expected cap height cap height (baseline to top of capital letters) as a scale
  capHeightScale?: number;

  // Sets the line width for underlines
  underlineLineWidth?: number;

  // Sets the underline height as a scale relative to text bounds height
  underlineHeightScale?: number;

  // Sets line width for strikethrough
  strikethroughLineWidth?: number;

  // Sets height of strikethrough as a scale relative to text bounds height
  strikethroughHeightScale?: number;

  // Sets the fill for links within the text
  linkFill?: TPaint;

  // Sets whether link clicks will call event.handle()
  linkEventsHandled?: boolean;

  // Sets the map of href placeholder => actual href/callback used for links
  links?: RichTextLinks;

  // Sets text alignment if there are multiple lines
  align?: RichTextAlign;

  // Sets the spacing between lines if there are multiple lines
  leading?: number;

  // Sets width of text before creating a new line
  lineWrap?: number | null;

  // Sets forwarding of the textProperty, see setTextProperty() for more documentation
  textProperty?: TReadOnlyProperty<string> | null;

  textPropertyOptions?: PropertyOptions<string>;

  // Sets the text to be displayed by this Node
  text?: string | number;
};

export type RichTextOptions = SelfOptions & NodeOptions;

type HimalayaAttribute = {
  key: string;
  value?: string;
};
type HimalayaNode = {
  type: 'element' | 'comment' | 'text';
  innerContent: string;
};
type HimalayaElementNode = {
  type: 'element';
  tagName: string;
  children: HimalayaNode[];
  attributes: HimalayaAttribute[];
  innerContent?: string; // Is this in the generated stuff? Do we just override this? Unclear
} & HimalayaNode;
const isElementNode = ( node: HimalayaNode ): node is HimalayaElementNode => node.type.toLowerCase() === 'element';
type HimalayaTextNode = {
  type: 'text';
  content: string;
} & HimalayaNode;
const isTextNode = ( node: HimalayaNode ): node is HimalayaTextNode => node.type.toLowerCase() === 'text';

const DEFAULT_FONT = new Font( {
  size: 20
} );

const TEXT_PROPERTY_TANDEM_NAME = 'textProperty';

// Tags that should be included in accessible innerContent, see https://github.com/phetsims/joist/issues/430
const ACCESSIBLE_TAGS = [
  'b', 'strong', 'i', 'em', 'sub', 'sup', 'u', 's'
];

// What type of line-break situations we can be in during our recursive process
const LineBreakState = {
  // There was a line break, but it was at the end of the element (or was a <br>). The relevant element can be fully
  // removed from the tree.
  COMPLETE: 'COMPLETE',

  // There was a line break, but there is some content left in this element after the line break. DO NOT remove it.
  INCOMPLETE: 'INCOMPLETE',

  // There was NO line break
  NONE: 'NONE'
};

/**
 * Get the attribute value from an element. Return null if that attribute isn't on the element.
 */
const himalayaGetAttribute = ( attribute: string, element: HimalayaElementNode | null ): string | null => {
  if ( !element ) {
    return null;
  }
  const attributeObject = _.find( element.attributes, x => x.key === attribute );
  if ( !attributeObject ) {
    return null;
  }
  return attributeObject.value || null;
};

/**
 * Turn a string of style like "font-sie:6; font-weight:6; favorite-number:6" into a may of style key/values (trimmed of whitespace)
 */
const himalayaStyleStringToMap = ( styleString: string ): Record<string, string> => {
  const styleElements = styleString.split( ';' );
  const styleMap: Record<string, string> = {};
  styleElements.forEach( styleKeyValue => {
    if ( styleKeyValue.length > 0 ) {
      const keyValueTuple = styleKeyValue.split( ':' );
      assert && assert( keyValueTuple.length === 2, 'too many colons' );
      styleMap[ keyValueTuple[ 0 ].trim() ] = keyValueTuple[ 1 ].trim();
    }
  } );
  return styleMap;
};

// We need to do some font-size tests, so we have a Text for that.
const scratchText = new Text( '' );

// himalaya converts dash separated CSS to camel case - use CSS compatible style with dashes, see above for examples
const FONT_STYLE_MAP = {
  'font-family': 'family',
  'font-size': 'size',
  'font-stretch': 'stretch',
  'font-style': 'style',
  'font-variant': 'variant',
  'font-weight': 'weight',
  'line-height': 'lineHeight'
} as const;

const FONT_STYLE_KEYS = Object.keys( FONT_STYLE_MAP );
const STYLE_KEYS = [ 'color' ].concat( FONT_STYLE_KEYS );

export default class RichText extends Node {

  // The text to display. We'll initialize this by mutating.
  private readonly _textProperty: TinyForwardingProperty<string>;

  private _font: Font | string;
  private _boundsMethod: TextBoundsMethod;
  private _fill: TPaint;
  private _stroke: TPaint;
  private _lineWidth: number;

  private _subScale: number;
  private _subXSpacing: number;
  private _subYOffset: number;

  private _supScale: number;
  private _supXSpacing: number;
  private _supYOffset: number;

  private _capHeightScale: number;

  private _underlineLineWidth: number;
  private _underlineHeightScale: number;

  private _strikethroughLineWidth: number;
  private _strikethroughHeightScale: number;

  private _linkFill: TPaint;

  private _linkEventsHandled: boolean;

  // If an object, values are either {string} or {function}
  private _links: Record<string, string> | boolean;

  private _align: RichTextAlign;
  private _leading: number;
  private _lineWrap: number | null;

  // We need to consolidate links (that could be split across multiple lines) under one "link" node, so we track created
  // link fragments here so they can get pieced together later.
  private _linkItems: { element: HimalayaNode; node: Node; href: string }[];

  // Whether something has been added to this line yet. We don't want to infinite-loop out if something is longer than
  // our lineWrap, so we'll place one item on its own on an otherwise empty line.
  private _hasAddedLeafToLine: boolean;

  // Normal layout container of lines (separate, so we can clear it easily)
  private lineContainer: Node;

  public constructor( text: string | number | TReadOnlyProperty<string>, providedOptions?: RichTextOptions ) {

    // We only fill in some defaults, since the other defaults are defined below (and mutate is relied on)
    const options = optionize<RichTextOptions, Pick<SelfOptions, 'fill'>, NodeOptions>()( {
      fill: '#000000',
      tandem: Tandem.OPTIONAL,
      phetioType: RichText.RichTextIO
    }, providedOptions );

    if ( typeof text === 'string' || typeof text === 'number' ) {
      options.text = text;
    }
    else {
      options.textProperty = text;
    }

    super();

    this._textProperty = new TinyForwardingProperty( '', true, this.onTextPropertyChange.bind( this ) );
    this._font = DEFAULT_FONT;
    this._boundsMethod = 'hybrid';
    this._fill = '#000000';
    this._stroke = null;
    this._lineWidth = 1;
    this._subScale = 0.75;
    this._subXSpacing = 0;
    this._subYOffset = 0;
    this._supScale = 0.75;
    this._supXSpacing = 0;
    this._supYOffset = 0;
    this._capHeightScale = 0.75;
    this._underlineLineWidth = 1;
    this._underlineHeightScale = 0.15;
    this._strikethroughLineWidth = 1;
    this._strikethroughHeightScale = 0.3;
    this._linkFill = 'rgb(27,0,241)';
    this._linkEventsHandled = false;
    this._links = {};
    this._align = 'left';
    this._leading = 0;
    this._lineWrap = null;
    this._linkItems = [];
    this._hasAddedLeafToLine = false;

    this.lineContainer = new Node( {} );
    this.addChild( this.lineContainer );

    // Initialize to an empty state, so we are immediately valid (since now we need to create an empty leaf even if we
    // have empty text).
    this.rebuildRichText();

    this.mutate( options );
  }

  /**
   * Called when our text Property changes values.
   */
  private onTextPropertyChange(): void {
    this.rebuildRichText();
  }

  /**
   * See documentation for Node.setVisibleProperty, except this is for the text string.
   */
  public setTextProperty( newTarget: TProperty<string> | null ): this {
    return this._textProperty.setTargetProperty( this, TEXT_PROPERTY_TANDEM_NAME, newTarget );
  }

  public set textProperty( property: TProperty<string> | null ) { this.setTextProperty( property ); }

  public get textProperty(): TProperty<string> { return this.getTextProperty(); }

  /**
   * Like Node.getVisibleProperty, but for the text string. Note this is not the same as the Property provided in
   * setTextProperty. Thus is the nature of TinyForwardingProperty.
   */
  public getTextProperty(): TProperty<string> {
    return this._textProperty;
  }

  /**
   * See documentation and comments in Node.initializePhetioObject
   */
  public override initializePhetioObject( baseOptions: Partial<PhetioObjectOptions>, providedOptions: RichTextOptions ): void {

    const options = optionize<RichTextOptions, EmptySelfOptions, RichTextOptions>()( {}, providedOptions );

    // Track this, so we only override our textProperty once.
    const wasInstrumented = this.isPhetioInstrumented();

    super.initializePhetioObject( baseOptions, options );

    if ( Tandem.PHET_IO_ENABLED && !wasInstrumented && this.isPhetioInstrumented() ) {

      this._textProperty.initializePhetio( this, TEXT_PROPERTY_TANDEM_NAME, () => {
        return new StringProperty( this.text, combineOptions<RichTextOptions>( {

          // by default, use the value from the Node
          phetioReadOnly: this.phetioReadOnly,
          tandem: this.tandem.createTandem( TEXT_PROPERTY_TANDEM_NAME ),
          phetioDocumentation: 'Property for the displayed text'
        }, options.textPropertyOptions ) );
      } );
    }
  }

  /**
   * When called, will rebuild the node structure for this RichText
   */
  private rebuildRichText(): void {
    this.freeChildrenToPool();

    // Bail early, particularly if we are being constructed.
    if ( this.text === '' ) {
      this.appendEmptyLeaf();
      return;
    }

    sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `RichText#${this.id} rebuild` );
    sceneryLog && sceneryLog.RichText && sceneryLog.push();

    // Turn bidirectional marks into explicit elements, so that the nesting is applied correctly.
    const mappedText = this.text.replace( /\u202a/g, '<span dir="ltr">' )
      .replace( /\u202b/g, '<span dir="rtl">' )
      .replace( /\u202c/g, '</span>' );

    // Start appending all top-level elements
    // @ts-ignore
    const rootElements: HimalayaNode[] = himalaya.parse( mappedText );

    // Clear out link items, as we'll need to reconstruct them later
    this._linkItems.length = 0;

    const widthAvailable = this._lineWrap === null ? Number.POSITIVE_INFINITY : this._lineWrap;
    const isRootLTR = true;

    let currentLine = RichTextElement.pool.create( isRootLTR );
    this._hasAddedLeafToLine = false; // notify that if nothing has been added, the first leaf always gets added.

    // Himalaya can give us multiple top-level items, so we need to iterate over those
    while ( rootElements.length ) {
      const element = rootElements[ 0 ];

      // How long our current line is already
      const currentLineWidth = currentLine.bounds.isValid() ? currentLine.width : 0;

      // Add the element in
      const lineBreakState = this.appendElement( currentLine, element, this._font, this._fill, isRootLTR, widthAvailable - currentLineWidth );
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `lineBreakState: ${lineBreakState}` );

      // If there was a line break (we'll need to swap to a new line node)
      if ( lineBreakState !== LineBreakState.NONE ) {
        // Add the line if it works
        if ( currentLine.bounds.isValid() ) {
          sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Adding line due to lineBreak' );
          this.appendLine( currentLine );
        }
        // Otherwise if it's a blank line, add in a strut (<br><br> should result in a blank line)
        else {
          this.appendLine( new VStrut( scratchText.setText( 'X' ).setFont( this._font ).height ) );
        }

        // Set up a new line
        currentLine = RichTextElement.pool.create( isRootLTR );
        this._hasAddedLeafToLine = false;
      }

      // If it's COMPLETE or NONE, then we fully processed the line
      if ( lineBreakState !== LineBreakState.INCOMPLETE ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Finished root element' );
        rootElements.splice( 0, 1 );
      }
    }

    // Only add the final line if it's valid (we don't want to add unnecessary padding at the bottom)
    if ( currentLine.bounds.isValid() ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Adding final line' );
      this.appendLine( currentLine );
    }

    // If we reached here and have no children, we probably ran into a degenerate "no layout" case like `' '`. Add in
    // the empty leaf.
    if ( this.lineContainer.getChildrenCount() === 0 ) {
      this.appendEmptyLeaf();
    }

    // All lines are constructed, so we can align them now
    this.alignLines();

    // Handle regrouping of links (so that all fragments of a link across multiple lines are contained under a single
    // ancestor that has listeners and a11y)
    while ( this._linkItems.length ) {
      // Close over the href and other references
      ( () => {
        const linkElement = this._linkItems[ 0 ].element;
        const href = this._linkItems[ 0 ].href;
        let i;

        // Find all nodes that are for the same link
        const nodes = [];
        for ( i = this._linkItems.length - 1; i >= 0; i-- ) {
          const item = this._linkItems[ i ];
          if ( item.element === linkElement ) {
            nodes.push( item.node );
            this._linkItems.splice( i, 1 );
          }
        }

        const linkRootNode = RichTextLink.pool.create( linkElement.innerContent, href );
        this.lineContainer.addChild( linkRootNode );

        // Detach the node from its location, adjust its transform, and reattach under the link. This should keep each
        // fragment in the same place, but changes its parent.
        for ( i = 0; i < nodes.length; i++ ) {
          const node = nodes[ i ];
          const matrix = node.getUniqueTrailTo( this.lineContainer ).getMatrix();
          node.detach();
          node.matrix = matrix;
          linkRootNode.addChild( node );
        }
      } )();
    }

    // Clear them out afterwards, for memory purposes
    this._linkItems.length = 0;

    sceneryLog && sceneryLog.RichText && sceneryLog.pop();
  }

  /**
   * Cleans "recursively temporary disposes" the children.
   */
  private freeChildrenToPool(): void {
    // Clear any existing lines or link fragments (higher performance, and return them to pools also)
    while ( this.lineContainer._children.length ) {
      const child = this.lineContainer._children[ this.lineContainer._children.length - 1 ] as RichTextCleanableNode;
      this.lineContainer.removeChild( child );

      //TODO https://github.com/phetsims/scenery/issues/1444 temporary workaround
      child.clean && child.clean();
    }
  }

  /**
   * Releases references.
   */
  public override dispose(): void {
    this.freeChildrenToPool();

    super.dispose();

    this._textProperty.dispose();
  }

  /**
   * Appends a finished line, applying any necessary leading.
   */
  private appendLine( lineNode: RichTextElement | Node ): void {
    // Apply leading
    if ( this.lineContainer.bounds.isValid() ) {
      lineNode.top = this.lineContainer.bottom + this._leading;

      // This ensures RTL lines will still be laid out properly with the main origin (handled by alignLines later)
      lineNode.left = 0;
    }

    this.lineContainer.addChild( lineNode );
  }

  /**
   * If we end up with the equivalent of "no" content, toss in a basically empty leaf so that we get valid bounds
   * (0 width, correctly-positioned height). See https://github.com/phetsims/scenery/issues/769.
   */
  private appendEmptyLeaf(): void {
    assert && assert( this.lineContainer.getChildrenCount() === 0 );

    this.appendLine( RichTextLeaf.pool.create( '', true, this._font, this._boundsMethod, this._fill, this._stroke, this._lineWidth ) );
  }

  /**
   * Aligns all lines attached to the lineContainer.
   */
  private alignLines(): void {
    // All nodes will either share a 'left', 'centerX' or 'right'.
    const coordinateName = this._align === 'center' ? 'centerX' : this._align;

    const ideal = this.lineContainer[ coordinateName ];
    for ( let i = 0; i < this.lineContainer.getChildrenCount(); i++ ) {
      this.lineContainer.getChildAt( i )[ coordinateName ] = ideal;
    }
  }

  /**
   * Main recursive function for constructing the RichText Node tree.
   *
   * We'll add any relevant content to the containerNode. The element will be mutated as things are added, so that
   * whenever content is added to the Node tree it will be removed from the element tree. This means we can pause
   * whenever (e.g. when a line-break is encountered) and the rest will be ready for parsing the next line.
   *
   * @param containerNode - The node where child elements should be placed
   * @param element - See Himalaya's element specification
   *                      (https://github.com/andrejewski/himalaya/blob/master/text/ast-spec-v0.md)
   * @param font - The font to apply at this level
   * @param fill - Fill to apply
   * @param isLTR - True if LTR, false if RTL (handles RTL text properly)
   * @param widthAvailable - How much width we have available before forcing a line break (for lineWrap)
   * @returns - Whether a line break was reached
   */
  private appendElement( containerNode: RichTextElement, element: HimalayaNode, font: Font | string, fill: TPaint, isLTR: boolean, widthAvailable: number ): string {
    let lineBreakState = LineBreakState.NONE;

    // {Node|Text} - The main Node for the element that we are adding
    let node!: Node | Text;

    // If we're a leaf
    if ( isTextNode( element ) ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `appending leaf: ${element.content}` );
      sceneryLog && sceneryLog.RichText && sceneryLog.push();

      node = RichTextLeaf.pool.create( element.content, isLTR, font, this._boundsMethod, fill, this._stroke, this._lineWidth );

      // If this content gets added, it will need to be pushed over by this amount
      const containerSpacing = isLTR ? containerNode.rightSpacing : containerNode.leftSpacing;

      // Handle wrapping if required. Container spacing cuts into our available width
      if ( !( node as RichTextLeaf ).fitsIn( widthAvailable - containerSpacing, this._hasAddedLeafToLine, isLTR ) ) {
        // Didn't fit, lets break into words to see what we can fit. We'll create ranges for all the individual
        // elements we could split the lines into. If we split into different lines, we can ignore the characters
        // in-between, however if not, we need to include them.
        const ranges = getLineBreakRanges( element.content );

        // Convert a group of ranges into a string (grab the content from the string).
        const rangesToString = ( ranges: Range[] ): string => {
          if ( ranges.length === 0 ) {
            return '';
          }
          else {
            return element.content.slice( ranges[ 0 ].min, ranges[ ranges.length - 1 ].max );
          }
        };

        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `Overflow leafAdded:${this._hasAddedLeafToLine}, words: ${ranges.length}` );

        // If we need to add something (and there is only a single word), then add it
        if ( this._hasAddedLeafToLine || ranges.length > 1 ) {
          sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Skipping words' );

          const skippedRanges: Range[] = [];
          let success = false;
          skippedRanges.unshift( ranges.pop()! ); // We didn't fit with the last one!

          // Keep shortening by removing words until it fits (or if we NEED to fit it) or it doesn't fit.
          while ( ranges.length ) {
            node = RichTextLeaf.pool.create( rangesToString( ranges ), isLTR, font, this._boundsMethod, fill, this._stroke, this._lineWidth );

            // If we haven't added anything to the line and we are down to the first word, we need to just add it.
            if ( !( node as RichTextLeaf ).fitsIn( widthAvailable - containerSpacing, this._hasAddedLeafToLine, isLTR ) &&
                 ( this._hasAddedLeafToLine || ranges.length > 1 ) ) {
              sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `Skipping word ${rangesToString( [ ranges[ ranges.length - 1 ] ] )}` );
              skippedRanges.unshift( ranges.pop()! );
            }
            else {
              sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `Success with ${rangesToString( ranges )}` );
              success = true;
              break;
            }
          }

          // If we haven't added anything yet to this line, we'll permit the overflow
          if ( success ) {
            lineBreakState = LineBreakState.INCOMPLETE;
            element.content = rangesToString( skippedRanges );
            sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `Remaining content: ${element.content}` );
          }
          else {
            return LineBreakState.INCOMPLETE;
          }
        }
      }

      this._hasAddedLeafToLine = true;

      sceneryLog && sceneryLog.RichText && sceneryLog.pop();
    }
    // Otherwise presumably an element with content
    else if ( isElementNode( element ) ) {
      // Bail out quickly for a line break
      if ( element.tagName === 'br' ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'manual line break' );
        return LineBreakState.COMPLETE;
      }
      // Span (dir attribute) -- we need the LTR/RTL knowledge before most other operations
      else if ( element.tagName === 'span' ) {
        const dirAttributeString = himalayaGetAttribute( 'dir', element );

        if ( dirAttributeString ) {
          assert && assert( dirAttributeString === 'ltr' || dirAttributeString === 'rtl',
            'Span dir attributes should be ltr or rtl.' );
          isLTR = dirAttributeString === 'ltr';
        }
      }

      node = RichTextElement.pool.create( isLTR );

      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'appending element' );
      sceneryLog && sceneryLog.RichText && sceneryLog.push();

      const styleAttributeString = himalayaGetAttribute( 'style', element );

      if ( styleAttributeString ) {
        const css = himalayaStyleStringToMap( styleAttributeString );
        assert && Object.keys( css ).forEach( key => {
          assert!( _.includes( STYLE_KEYS, key ), 'See supported style CSS keys' );
        } );

        // Fill
        if ( css.color ) {
          fill = new Color( css.color );
        }

        // Font
        const fontOptions: Record<string, string> = {};
        for ( let i = 0; i < FONT_STYLE_KEYS.length; i++ ) {
          const styleKey = FONT_STYLE_KEYS[ i ];
          if ( css[ styleKey ] ) {
            // @ts-ignore
            fontOptions[ FONT_STYLE_MAP[ styleKey ] ] = css[ styleKey ];
          }
        }
        font = ( typeof font === 'string' ? Font.fromCSS( font ) : font ).copy( fontOptions );
      }

      // Achor (link)
      if ( element.tagName === 'a' ) {
        let href = himalayaGetAttribute( 'href', element );

        // Try extracting the href from the links object
        if ( href !== null && this._links !== true ) {
          if ( href.startsWith( '{{' ) && href.indexOf( '}}' ) === href.length - 2 ) {
            // @ts-ignore TODO
            href = this._links[ href.slice( 2, -2 ) ];
          }
          else {
            href = null;
          }
        }

        // Ignore things if there is no matching href
        if ( href ) {
          if ( this._linkFill !== null ) {
            fill = this._linkFill; // Link color
          }
          // Don't overwrite only innerContents once things have been "torn down"
          if ( !element.innerContent ) {
            element.innerContent = RichText.himalayaElementToAccessibleString( element, isLTR );
          }

          // Store information about it for the "regroup links" step
          this._linkItems.push( {
            element: element,
            node: node,
            href: href
          } );
        }
      }
      // Bold
      else if ( element.tagName === 'b' || element.tagName === 'strong' ) {
        font = ( typeof font === 'string' ? Font.fromCSS( font ) : font ).copy( {
          weight: 'bold'
        } );
      }
      // Italic
      else if ( element.tagName === 'i' || element.tagName === 'em' ) {
        font = ( typeof font === 'string' ? Font.fromCSS( font ) : font ).copy( {
          style: 'italic'
        } );
      }
      // Subscript
      else if ( element.tagName === 'sub' ) {
        node.scale( this._subScale );
        ( node as RichTextElement ).addExtraBeforeSpacing( this._subXSpacing );
        node.y += this._subYOffset;
      }
      // Superscript
      else if ( element.tagName === 'sup' ) {
        node.scale( this._supScale );
        ( node as RichTextElement ).addExtraBeforeSpacing( this._supXSpacing );
        node.y += this._supYOffset;
      }

      // If we've added extra spacing, we'll need to subtract it off of our available width
      const scale = node.getScaleVector().x;

      // Process children
      while ( lineBreakState === LineBreakState.NONE && element.children.length ) {
        const widthBefore = node.bounds.isValid() ? node.width : 0;

        const childElement = element.children[ 0 ];
        lineBreakState = this.appendElement( node as RichTextElement, childElement, font, fill, isLTR, widthAvailable / scale );

        // for COMPLETE or NONE, we'll want to remove the childElement from the tree (we fully processed it)
        if ( lineBreakState !== LineBreakState.INCOMPLETE ) {
          element.children.splice( 0, 1 );
        }

        const widthAfter = node.bounds.isValid() ? node.width : 0;

        // Remove the amount of width taken up by the child
        widthAvailable += widthBefore - widthAfter;
      }
      // If there is a line break and there are still more things to process, we are incomplete
      if ( lineBreakState === LineBreakState.COMPLETE && element.children.length ) {
        lineBreakState = LineBreakState.INCOMPLETE;
      }

      // Subscript positioning
      if ( element.tagName === 'sub' ) {
        if ( isFinite( node.height ) ) {
          node.centerY = 0;
        }
      }
      // Superscript positioning
      else if ( element.tagName === 'sup' ) {
        if ( isFinite( node.height ) ) {
          node.centerY = scratchText.setText( 'X' ).setFont( font ).top * this._capHeightScale;
        }
      }
      // Underline
      else if ( element.tagName === 'u' ) {
        const underlineY = -node.top * this._underlineHeightScale;
        if ( isFinite( node.top ) ) {
          node.addChild( new Line( node.localLeft, underlineY, node.localRight, underlineY, {
            stroke: fill,
            lineWidth: this._underlineLineWidth
          } ) );
        }
      }
      // Strikethrough
      else if ( element.tagName === 's' ) {
        const strikethroughY = node.top * this._strikethroughHeightScale;
        if ( isFinite( node.top ) ) {
          node.addChild( new Line( node.localLeft, strikethroughY, node.localRight, strikethroughY, {
            stroke: fill,
            lineWidth: this._strikethroughLineWidth
          } ) );
        }
      }
      sceneryLog && sceneryLog.RichText && sceneryLog.pop();
    }

    if ( node ) {
      const wasAdded = containerNode.addElement( node as RichTextElement | RichTextLeaf );
      if ( !wasAdded ) {
        // Remove it from the linkItems if we didn't actually add it.
        this._linkItems = this._linkItems.filter( item => item.node !== node );

        // And since we won't dispose it (since it's not a child), clean it here
        ( node as RichTextCleanableNode ).clean();
      }
    }

    return lineBreakState;
  }

  /**
   * Sets the text displayed by our node.
   *
   * NOTE: Encoding HTML entities is required, and malformed HTML is not accepted.
   *
   * @param text - The text to display. If it's a number, it will be cast to a string
   */
  public setText( text: string | number ): this {
    assert && assert( text !== null && text !== undefined, 'Text should be defined and non-null. Use the empty string if needed.' );
    assert && assert( typeof text === 'number' || typeof text === 'string', 'text should be a string or number' );

    // cast it to a string (for numbers, etc., and do it before the change guard so we don't accidentally trigger on non-changed text)
    text = `${text}`;

    this._textProperty.set( text );

    return this;
  }

  public set text( value: string | number ) { this.setText( value ); }

  public get text(): string { return this.getText(); }

  /**
   * Returns the text displayed by our node.
   */
  public getText(): string {
    return this._textProperty.value;
  }

  /**
   * Sets the method that is used to determine bounds from the text. See Text.setBoundsMethod for details
   */
  public setBoundsMethod( method: TextBoundsMethod ): this {
    assert && assert( method === 'fast' || method === 'fastCanvas' || method === 'accurate' || method === 'hybrid', 'Unknown Text boundsMethod' );
    if ( method !== this._boundsMethod ) {
      this._boundsMethod = method;
      this.rebuildRichText();
    }
    return this;
  }

  public set boundsMethod( value: TextBoundsMethod ) { this.setBoundsMethod( value ); }

  public get boundsMethod(): TextBoundsMethod { return this.getBoundsMethod(); }

  /**
   * Returns the current method to estimate the bounds of the text. See setBoundsMethod() for more information.
   */
  public getBoundsMethod(): TextBoundsMethod {
    return this._boundsMethod;
  }

  /**
   * Sets the font of our node.
   */
  public setFont( font: Font | string ): this {
    assert && assert( font instanceof Font || typeof font === 'string',
      'Fonts provided to setFont should be a Font object or a string in the CSS3 font shortcut format' );

    if ( this._font !== font ) {
      this._font = font;
      this.rebuildRichText();
    }
    return this;
  }

  public set font( value: Font | string ) { this.setFont( value ); }

  public get font(): Font | string { return this.getFont(); }

  /**
   * Returns the current Font
   */
  public getFont(): Font | string {
    return this._font;
  }

  /**
   * Sets the fill of our text.
   */
  public setFill( fill: TPaint ): this {
    if ( this._fill !== fill ) {
      this._fill = fill;
      this.rebuildRichText();
    }
    return this;
  }

  public set fill( value: TPaint ) { this.setFill( value ); }

  public get fill(): TPaint { return this.getFill(); }

  /**
   * Returns the current fill.
   */
  public getFill(): TPaint {
    return this._fill;
  }

  /**
   * Sets the stroke of our text.
   */
  public setStroke( stroke: TPaint ): this {
    if ( this._stroke !== stroke ) {
      this._stroke = stroke;
      this.rebuildRichText();
    }
    return this;
  }

  public set stroke( value: TPaint ) { this.setStroke( value ); }

  public get stroke(): TPaint { return this.getStroke(); }

  /**
   * Returns the current stroke.
   */
  public getStroke(): TPaint {
    return this._stroke;
  }

  /**
   * Sets the lineWidth of our text.
   */
  public setLineWidth( lineWidth: number ): this {
    if ( this._lineWidth !== lineWidth ) {
      this._lineWidth = lineWidth;
      this.rebuildRichText();
    }
    return this;
  }

  public set lineWidth( value: number ) { this.setLineWidth( value ); }

  public get lineWidth(): number { return this.getLineWidth(); }

  /**
   * Returns the current lineWidth.
   */
  public getLineWidth(): number {
    return this._lineWidth;
  }

  /**
   * Sets the scale (relative to 1) of any text under subscript (<sub>) elements.
   */
  public setSubScale( subScale: number ): this {
    assert && assert( typeof subScale === 'number' && isFinite( subScale ) && subScale > 0 );

    if ( this._subScale !== subScale ) {
      this._subScale = subScale;
      this.rebuildRichText();
    }
    return this;
  }

  public set subScale( value: number ) { this.setSubScale( value ); }

  public get subScale(): number { return this.getSubScale(); }

  /**
   * Returns the scale (relative to 1) of any text under subscript (<sub>) elements.
   */
  public getSubScale(): number {
    return this._subScale;
  }

  /**
   * Sets the horizontal spacing before any subscript (<sub>) elements.
   */
  public setSubXSpacing( subXSpacing: number ): this {
    assert && assert( typeof subXSpacing === 'number' && isFinite( subXSpacing ) );

    if ( this._subXSpacing !== subXSpacing ) {
      this._subXSpacing = subXSpacing;
      this.rebuildRichText();
    }
    return this;
  }

  public set subXSpacing( value: number ) { this.setSubXSpacing( value ); }

  public get subXSpacing(): number { return this.getSubXSpacing(); }

  /**
   * Returns the horizontal spacing before any subscript (<sub>) elements.
   */
  public getSubXSpacing(): number {
    return this._subXSpacing;
  }

  /**
   * Sets the adjustment offset to the vertical placement of any subscript (<sub>) elements.
   */
  public setSubYOffset( subYOffset: number ): this {
    assert && assert( typeof subYOffset === 'number' && isFinite( subYOffset ) );

    if ( this._subYOffset !== subYOffset ) {
      this._subYOffset = subYOffset;
      this.rebuildRichText();
    }
    return this;
  }

  public set subYOffset( value: number ) { this.setSubYOffset( value ); }

  public get subYOffset(): number { return this.getSubYOffset(); }

  /**
   * Returns the adjustment offset to the vertical placement of any subscript (<sub>) elements.
   */
  public getSubYOffset(): number {
    return this._subYOffset;
  }

  /**
   * Sets the scale (relative to 1) of any text under superscript (<sup>) elements.
   */
  public setSupScale( supScale: number ): this {
    assert && assert( typeof supScale === 'number' && isFinite( supScale ) && supScale > 0 );

    if ( this._supScale !== supScale ) {
      this._supScale = supScale;
      this.rebuildRichText();
    }
    return this;
  }

  public set supScale( value: number ) { this.setSupScale( value ); }

  public get supScale(): number { return this.getSupScale(); }

  /**
   * Returns the scale (relative to 1) of any text under superscript (<sup>) elements.
   */
  public getSupScale(): number {
    return this._supScale;
  }

  /**
   * Sets the horizontal spacing before any superscript (<sup>) elements.
   */
  public setSupXSpacing( supXSpacing: number ): this {
    assert && assert( typeof supXSpacing === 'number' && isFinite( supXSpacing ) );

    if ( this._supXSpacing !== supXSpacing ) {
      this._supXSpacing = supXSpacing;
      this.rebuildRichText();
    }
    return this;
  }

  public set supXSpacing( value: number ) { this.setSupXSpacing( value ); }

  public get supXSpacing(): number { return this.getSupXSpacing(); }

  /**
   * Returns the horizontal spacing before any superscript (<sup>) elements.
   */
  public getSupXSpacing(): number {
    return this._supXSpacing;
  }

  /**
   * Sets the adjustment offset to the vertical placement of any superscript (<sup>) elements.
   */
  public setSupYOffset( supYOffset: number ): this {
    assert && assert( typeof supYOffset === 'number' && isFinite( supYOffset ) );

    if ( this._supYOffset !== supYOffset ) {
      this._supYOffset = supYOffset;
      this.rebuildRichText();
    }
    return this;
  }

  public set supYOffset( value: number ) { this.setSupYOffset( value ); }

  public get supYOffset(): number { return this.getSupYOffset(); }

  /**
   * Returns the adjustment offset to the vertical placement of any superscript (<sup>) elements.
   */
  public getSupYOffset(): number {
    return this._supYOffset;
  }

  /**
   * Sets the expected cap height (baseline to top of capital letters) as a scale of the detected distance from the
   * baseline to the top of the text bounds.
   */
  public setCapHeightScale( capHeightScale: number ): this {
    assert && assert( typeof capHeightScale === 'number' && isFinite( capHeightScale ) && capHeightScale > 0 );

    if ( this._capHeightScale !== capHeightScale ) {
      this._capHeightScale = capHeightScale;
      this.rebuildRichText();
    }
    return this;
  }

  public set capHeightScale( value: number ) { this.setCapHeightScale( value ); }

  public get capHeightScale(): number { return this.getCapHeightScale(); }

  /**
   * Returns the expected cap height (baseline to top of capital letters) as a scale of the detected distance from the
   * baseline to the top of the text bounds.
   */
  public getCapHeightScale(): number {
    return this._capHeightScale;
  }

  /**
   * Sets the lineWidth of underline lines.
   */
  public setUnderlineLineWidth( underlineLineWidth: number ): this {
    assert && assert( typeof underlineLineWidth === 'number' && isFinite( underlineLineWidth ) && underlineLineWidth > 0 );

    if ( this._underlineLineWidth !== underlineLineWidth ) {
      this._underlineLineWidth = underlineLineWidth;
      this.rebuildRichText();
    }
    return this;
  }

  public set underlineLineWidth( value: number ) { this.setUnderlineLineWidth( value ); }

  public get underlineLineWidth(): number { return this.getUnderlineLineWidth(); }

  /**
   * Returns the lineWidth of underline lines.
   */
  public getUnderlineLineWidth(): number {
    return this._underlineLineWidth;
  }

  /**
   * Sets the underline height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   */
  public setUnderlineHeightScale( underlineHeightScale: number ): this {
    assert && assert( typeof underlineHeightScale === 'number' && isFinite( underlineHeightScale ) && underlineHeightScale > 0 );

    if ( this._underlineHeightScale !== underlineHeightScale ) {
      this._underlineHeightScale = underlineHeightScale;
      this.rebuildRichText();
    }
    return this;
  }

  public set underlineHeightScale( value: number ) { this.setUnderlineHeightScale( value ); }

  public get underlineHeightScale(): number { return this.getUnderlineHeightScale(); }

  /**
   * Returns the underline height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   */
  public getUnderlineHeightScale(): number {
    return this._underlineHeightScale;
  }

  /**
   * Sets the lineWidth of strikethrough lines.
   */
  public setStrikethroughLineWidth( strikethroughLineWidth: number ): this {
    assert && assert( typeof strikethroughLineWidth === 'number' && isFinite( strikethroughLineWidth ) && strikethroughLineWidth > 0 );

    if ( this._strikethroughLineWidth !== strikethroughLineWidth ) {
      this._strikethroughLineWidth = strikethroughLineWidth;
      this.rebuildRichText();
    }
    return this;
  }

  public set strikethroughLineWidth( value: number ) { this.setStrikethroughLineWidth( value ); }

  public get strikethroughLineWidth(): number { return this.getStrikethroughLineWidth(); }

  /**
   * Returns the lineWidth of strikethrough lines.
   */
  public getStrikethroughLineWidth(): number {
    return this._strikethroughLineWidth;
  }

  /**
   * Sets the strikethrough height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   */
  public setStrikethroughHeightScale( strikethroughHeightScale: number ): this {
    assert && assert( typeof strikethroughHeightScale === 'number' && isFinite( strikethroughHeightScale ) && strikethroughHeightScale > 0 );

    if ( this._strikethroughHeightScale !== strikethroughHeightScale ) {
      this._strikethroughHeightScale = strikethroughHeightScale;
      this.rebuildRichText();
    }
    return this;
  }

  public set strikethroughHeightScale( value: number ) { this.setStrikethroughHeightScale( value ); }

  public get strikethroughHeightScale(): number { return this.getStrikethroughHeightScale(); }

  /**
   * Returns the strikethrough height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   */
  public getStrikethroughHeightScale(): number {
    return this._strikethroughHeightScale;
  }

  /**
   * Sets the color of links. If null, no fill will be overridden.
   */
  public setLinkFill( linkFill: TPaint ): this {
    if ( this._linkFill !== linkFill ) {
      this._linkFill = linkFill;
      this.rebuildRichText();
    }
    return this;
  }

  public set linkFill( value: TPaint ) { this.setLinkFill( value ); }

  public get linkFill(): TPaint { return this.getLinkFill(); }

  /**
   * Returns the color of links.
   */
  public getLinkFill(): TPaint {
    return this._linkFill;
  }

  /**
   * Sets whether link clicks will call event.handle().
   */
  public setLinkEventsHandled( linkEventsHandled: boolean ): this {
    if ( this._linkEventsHandled !== linkEventsHandled ) {
      this._linkEventsHandled = linkEventsHandled;
      this.rebuildRichText();
    }
    return this;
  }

  public set linkEventsHandled( value: boolean ) { this.setLinkEventsHandled( value ); }

  public get linkEventsHandled(): boolean { return this.getLinkEventsHandled(); }

  /**
   * Returns whether link events will be handled.
   */
  public getLinkEventsHandled(): boolean {
    return this._linkEventsHandled;
  }

  /**
   * Sets the map of href placeholder => actual href/callback used for links. However if set to true ({boolean}) as a
   * full object, links in the string will not be mapped, but will be directly added.
   *
   * For instance, the default is to map hrefs for security purposes:
   *
   * new RichText( '<a href="{{alink}}">content</a>', {
   *   links: {
   *     alink: 'https://phet.colorado.edu'
   *   }
   * } );
   *
   * But links with an href not matching will be ignored. This can be avoided by passing links: true to directly
   * embed links:
   *
   * new RichText( '<a href="https://phet.colorado.edu">content</a>', { links: true } );
   *
   * Callbacks (instead of a URL) are also supported, e.g.:
   *
   * new RichText( '<a href="{{acallback}}">content</a>', {
   *   links: {
   *     acallback: function() { console.log( 'clicked' ) }
   *   }
   * } );
   *
   * See https://github.com/phetsims/scenery-phet/issues/316 for more information.
   */
  public setLinks( links: RichTextLinks ): this {
    assert && assert( links !== false || Object.getPrototypeOf( links ) === Object.prototype );

    if ( this._links !== links ) {
      this._links = links;
      this.rebuildRichText();
    }
    return this;
  }

  public set links( value: RichTextLinks ) { this.setLinks( value ); }

  public get links(): RichTextLinks { return this.getLinks(); }

  /**
   * Returns whether link events will be handled.
   */
  public getLinks(): RichTextLinks {
    return this._links;
  }

  /**
   * Sets the alignment of text (only relevant if there are multiple lines).
   */
  public setAlign( align: RichTextAlign ): this {
    assert && assert( align === 'left' || align === 'center' || align === 'right' );

    if ( this._align !== align ) {
      this._align = align;
      this.rebuildRichText();
    }
    return this;
  }

  public set align( value: RichTextAlign ) { this.setAlign( value ); }

  public get align(): RichTextAlign { return this.getAlign(); }

  /**
   * Returns the current alignment of the text (only relevant if there are multiple lines).
   */
  public getAlign(): RichTextAlign {
    return this._align;
  }

  /**
   * Sets the leading (spacing between lines)
   */
  public setLeading( leading: number ): this {
    assert && assert( typeof leading === 'number' && isFinite( leading ) );

    if ( this._leading !== leading ) {
      this._leading = leading;
      this.rebuildRichText();
    }
    return this;
  }

  public set leading( value: number ) { this.setLeading( value ); }

  public get leading(): number { return this.getLeading(); }

  /**
   * Returns the leading (spacing between lines)
   */
  public getLeading(): number {
    return this._leading;
  }

  /**
   * Sets the line wrap width for the text (or null if none is desired). Lines longer than this length will wrap
   * automatically to the next line.
   *
   * @param lineWrap - If it's a number, it should be greater than 0.
   */
  public setLineWrap( lineWrap: number | null ): this {
    assert && assert( lineWrap === null || ( typeof lineWrap === 'number' && isFinite( lineWrap ) && lineWrap > 0 ) );

    if ( this._lineWrap !== lineWrap ) {
      this._lineWrap = lineWrap;
      this.rebuildRichText();
    }
    return this;
  }

  public set lineWrap( value: number | null ) { this.setLineWrap( value ); }

  public get lineWrap(): number | null { return this.getLineWrap(); }

  /**
   * Returns the line wrap width.
   */
  public getLineWrap(): number | null {
    return this._lineWrap;
  }

  public override mutate( options?: RichTextOptions ): this {
    if ( assert && options && options.hasOwnProperty( 'text' ) && options.hasOwnProperty( 'textProperty' ) && options.textProperty ) {
      assert && assert( options.textProperty.value === options.text, 'If both text and textProperty are provided, then values should match' );
    }

    return super.mutate( options );
  }

  /**
   * Returns a wrapped version of the string with a font specifier that uses the given font object.
   *
   * NOTE: Does an approximation of some font values (using <b> or <i>), and cannot force the lack of those if it is
   * included in bold/italic exterior tags.
   */
  public static stringWithFont( str: string, font: Font ): string {
    // TODO: ES6 string interpolation.
    return `${'<span style=\'' +
           'font-style: '}${font.style};` +
           `font-variant: ${font.variant};` +
           `font-weight: ${font.weight};` +
           `font-stretch: ${font.stretch};` +
           `font-size: ${font.size};` +
           `font-family: ${font.family};` +
           `line-height: ${font.lineHeight};` +
           `'>${str}</span>`;
  }

  /**
   * Stringifies an HTML subtree defined by the given element.
   */
  public static himalayaElementToString( element: HimalayaNode, isLTR: boolean ): string {
    if ( isTextNode( element ) ) {
      return RichText.contentToString( element.content, isLTR );
    }
    else if ( isElementNode( element ) ) {
      const dirAttributeString = himalayaGetAttribute( 'dir', element );

      if ( element.tagName === 'span' && dirAttributeString ) {
        isLTR = dirAttributeString === 'ltr';
      }

      // Process children
      return element.children.map( child => RichText.himalayaElementToString( child, isLTR ) ).join( '' );
    }
    else {
      return '';
    }
  }

  /**
   * Stringifies an HTML subtree defined by the given element, but removing certain tags that we don't need for
   * accessibility (like <a>, <span>, etc.), and adding in tags we do want (see ACCESSIBLE_TAGS).
   */
  public static himalayaElementToAccessibleString( element: HimalayaNode, isLTR: boolean ): string {
    if ( isTextNode( element ) ) {
      return RichText.contentToString( element.content, isLTR );
    }
    else if ( isElementNode( element ) ) {
      const dirAttribute = himalayaGetAttribute( 'dir', element );

      if ( element.tagName === 'span' && dirAttribute ) {
        isLTR = dirAttribute === 'ltr';
      }

      // Process children
      const content = element.children.map( child => RichText.himalayaElementToAccessibleString( child, isLTR ) ).join( '' );

      if ( _.includes( ACCESSIBLE_TAGS, element.tagName ) ) {
        return `<${element.tagName}>${content}</${element.tagName}>`;
      }
      else {
        return content;
      }
    }
    else {
      return '';
    }
  }

  /**
   * Takes the element.content from himalaya, unescapes HTML entities, and applies the proper directional tags.
   *
   * See https://github.com/phetsims/scenery-phet/issues/315
   */
  public static contentToString( content: string, isLTR: boolean ): string {
    // @ts-ignore - we should get a string from this
    const unescapedContent: string = he.decode( content );
    return isLTR ? ( `\u202a${unescapedContent}\u202c` ) : ( `\u202b${unescapedContent}\u202c` );
  }

  public static RichTextIO: IOType;
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
RichText.prototype._mutatorKeys = RICH_TEXT_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'RichText', RichText );

const RichTextCleanable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should mix Paintable' );

  return class RichTextCleanableMixin extends type {
    public get isCleanable(): boolean {
      return true;
    }

    /**
     * Releases references
     */
    public clean(): void {
      const thisNode = this as unknown as RichTextCleanableNode;

      // Remove all children (and recursively clean)
      for ( let i = thisNode._children.length - 1; i >= 0; i-- ) {
        const child = thisNode._children[ i ] as RichTextCleanableNode;

        if ( child.isCleanable ) {
          thisNode.removeChild( child );
          child.clean();
        }
      }

      thisNode.matrix = Matrix3.IDENTITY;

      thisNode.freeToPool();
    }
  };
} );
type RichTextCleanableNode = Node & { clean: () => void; isCleanable: boolean; freeToPool: () => void };

class RichTextElement extends RichTextCleanable( Node ) {

  private isLTR!: boolean;

  // The amount of local-coordinate spacing to apply on each side
  public leftSpacing!: number;
  public rightSpacing!: number;

  /**
   * A container of other RichText elements and leaves.
   *
   * @param isLTR - Whether this container will lay out elements in the left-to-right order (if false, will be
   *                          right-to-left).
   */
  public constructor( isLTR: boolean ) {
    super();

    this.initialize( isLTR );
  }

  /**
   * Sets up state
   */
  public initialize( isLTR: boolean ): this {
    this.isLTR = isLTR;
    this.leftSpacing = 0;
    this.rightSpacing = 0;

    return this;
  }

  /**
   * Adds a child element.
   *
   * @returns- Whether the item was actually added.
   */
  public addElement( element: RichTextElement | RichTextLeaf ): boolean {

    const hadChild = this.children.length > 0;
    const hasElement = element.width > 0;

    // May be at a different scale, which we need to handle
    const elementScale = element.getScaleVector().x;
    const leftElementSpacing = element.leftSpacing * elementScale;
    const rightElementSpacing = element.rightSpacing * elementScale;

    // If there is nothing, than no spacing should be handled
    if ( !hadChild && !hasElement ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'No child or element, ignoring' );
      return false;
    }
    else if ( !hadChild ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `First child, ltr:${this.isLTR}, spacing: ${this.isLTR ? rightElementSpacing : leftElementSpacing}` );
      if ( this.isLTR ) {
        element.left = 0;
        this.leftSpacing = leftElementSpacing;
        this.rightSpacing = rightElementSpacing;
      }
      else {
        element.right = 0;
        this.leftSpacing = leftElementSpacing;
        this.rightSpacing = rightElementSpacing;
      }
      this.addChild( element );
      return true;
    }
    else if ( !hasElement ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `No element, adding spacing, ltr:${this.isLTR}, spacing: ${leftElementSpacing + rightElementSpacing}` );
      if ( this.isLTR ) {
        this.rightSpacing += leftElementSpacing + rightElementSpacing;
      }
      else {
        this.leftSpacing += leftElementSpacing + rightElementSpacing;
      }
    }
    else {
      if ( this.isLTR ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `LTR add ${this.rightSpacing} + ${leftElementSpacing}` );
        element.left = this.localRight + this.rightSpacing + leftElementSpacing;
        this.rightSpacing = rightElementSpacing;
      }
      else {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `RTL add ${this.leftSpacing} + ${rightElementSpacing}` );
        element.right = this.localLeft - this.leftSpacing - rightElementSpacing;
        this.leftSpacing = leftElementSpacing;
      }
      this.addChild( element );
      return true;
    }
    return false;
  }

  /**
   * Adds an amount of spacing to the "before" side.
   */
  public addExtraBeforeSpacing( amount: number ): void {
    if ( this.isLTR ) {
      this.leftSpacing += amount;
    }
    else {
      this.rightSpacing += amount;
    }
  }

  public freeToPool(): void {
    RichTextElement.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextElement );

}

class RichTextLeaf extends RichTextCleanable( Text ) {

  public leftSpacing!: number;
  public rightSpacing!: number;

  /**
   * A leaf (text) node.
   */
  public constructor( content: string, isLTR: boolean, font: Font | string, boundsMethod: TextBoundsMethod, fill: TPaint, stroke: TPaint, lineWidth: number ) {
    super( '' );

    this.initialize( content, isLTR, font, boundsMethod, fill, stroke, lineWidth );
  }

  /**
   * Set up this text's state
   */
  public initialize( content: string, isLTR: boolean, font: Font | string, boundsMethod: TextBoundsMethod, fill: TPaint, stroke: TPaint, lineWidth: number ): this {

    // Grab all spaces at the (logical) start
    let whitespaceBefore = '';
    while ( content.startsWith( ' ' ) ) {
      whitespaceBefore += ' ';
      content = content.slice( 1 );
    }

    // Grab all spaces at the (logical) end
    let whitespaceAfter = '';
    while ( content.endsWith( ' ' ) ) {
      whitespaceAfter = ' ';
      content = content.slice( 0, content.length - 1 );
    }

    this.text = RichText.contentToString( content, isLTR );
    this.boundsMethod = boundsMethod;
    this.font = font;
    this.fill = fill;
    this.stroke = stroke;
    this.lineWidth = lineWidth;

    const spacingBefore = whitespaceBefore.length ? scratchText.setText( whitespaceBefore ).setFont( font ).width : 0;
    const spacingAfter = whitespaceAfter.length ? scratchText.setText( whitespaceAfter ).setFont( font ).width : 0;

    // Turn logical spacing into directional
    this.leftSpacing = isLTR ? spacingBefore : spacingAfter;
    this.rightSpacing = isLTR ? spacingAfter : spacingBefore;

    return this;
  }

  /**
   * Cleans references that could cause memory leaks (as those things may contain other references).
   */
  public override clean(): void {
    super.clean();

    this.fill = null;
    this.stroke = null;
  }

  /**
   * Whether this leaf will fit in the specified amount of space (including, if required, the amount of spacing on
   * the side).
   */
  public fitsIn( widthAvailable: number, hasAddedLeafToLine: boolean, isContainerLTR: boolean ): boolean {
    return this.width + ( hasAddedLeafToLine ? ( isContainerLTR ? this.leftSpacing : this.rightSpacing ) : 0 ) <= widthAvailable;
  }

  public freeToPool(): void {
    RichTextLeaf.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextLeaf );
}

class RichTextLink extends Voicing( RichTextCleanable( Node ) ) {

  private fireListener: FireListener | null;
  private accessibleInputListener: TInputListener | null;

  /**
   * A link node
   */
  public constructor( innerContent: string, href: RichTextHref ) {
    super();

    this.fireListener = null;
    this.accessibleInputListener = null;

    // Voicing was already initialized in the super call, we do not want to initialize super again. But we do want to
    // initialize the RichText portion of the implementation.
    this.initialize( innerContent, href, false );

    // Mutate to make sure initialize doesn't clear this away
    this.mutate( {
      cursor: 'pointer',
      tagName: 'a'
    } );
  }

  /**
   * Set up this state. First construction does not need to use super.initialize() because the constructor has done
   * that for us. But repeated initialization with Poolable will need to initialize super again.
   */
  public override initialize( innerContent: string, href: RichTextHref, initializeSuper = true ): this {

    if ( initializeSuper ) {
      super.initialize();
    }

    // pdom - open the link in the new tab when activated with a keyboard.
    // also see https://github.com/phetsims/joist/issues/430
    this.innerContent = innerContent;

    this.voicingNameResponse = innerContent;

    // If our href is a function, it should be called when the user clicks on the link
    if ( typeof href === 'function' ) {
      this.fireListener = new FireListener( {
        fire: href,
        tandem: Tandem.OPT_OUT
      } );
      this.addInputListener( this.fireListener );
      this.setPDOMAttribute( 'href', '#' ); // Required so that the click listener will get called.
      this.setPDOMAttribute( 'target', '_self' ); // This is the default (easier than conditionally removing)
      this.accessibleInputListener = {
        click: event => {
          event.domEvent && event.domEvent.preventDefault();

          href();
        }
      };
      this.addInputListener( this.accessibleInputListener );
    }
    // Otherwise our href is a {string}, and we should open a window pointing to it (assuming it's a URL)
    else {
      this.fireListener = new FireListener( {
        fire: event => {
          if ( event.isFromPDOM() ) {

            // prevent default from pdom activation so we don't also open a new tab from native DOM input on a link
            event.domEvent!.preventDefault();
          }
          // @ts-ignore TODO TODO TODO this is a bug! How do we handle this?
          self._linkEventsHandled && event.handle();
          openPopup( href ); // open in a new window/tab
        },
        tandem: Tandem.OPT_OUT
      } );
      this.addInputListener( this.fireListener );
      this.setPDOMAttribute( 'href', href );
      this.setPDOMAttribute( 'target', '_blank' );
    }

    return this;
  }

  /**
   * Cleans references that could cause memory leaks (as those things may contain other references).
   */
  public override clean(): void {
    super.clean();

    this.fireListener && this.removeInputListener( this.fireListener );
    this.fireListener = null;
    if ( this.accessibleInputListener ) {
      this.removeInputListener( this.accessibleInputListener );
      this.accessibleInputListener = null;
    }
  }

  public freeToPool(): void {
    RichTextLink.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextLink );
}

RichText.RichTextIO = new IOType( 'RichTextIO', {
  valueType: RichText,
  supertype: Node.NodeIO,
  documentation: 'The tandem IO Type for the scenery RichText node'
} );
