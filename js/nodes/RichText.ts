// Copyright 2017-2023, University of Colorado Boulder

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
 * - <node id="id"> for embedding a Node into the text (pass in { nodes: { id: NODE } }), with optional align attribute
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
import Tandem from '../../../tandem/js/Tandem.js';
import IOType from '../../../tandem/js/types/IOType.js';
import { allowLinksProperty, Color, Font, getLineBreakRanges, HimalayaNode, isHimalayaElementNode, isHimalayaTextNode, Line, Node, NodeOptions, RichTextCleanableNode, RichTextElement, RichTextLeaf, RichTextLink, RichTextNode, RichTextUtils, RichTextVerticalSpacer, scenery, Text, TextBoundsMethod, TPaint } from '../imports.js';
import optionize, { combineOptions, EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';

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
  'nodes',
  'replaceNewlines',
  'align',
  'leading',
  'lineWrap',
  Text.STRING_PROPERTY_NAME,
  'string'
];

export type RichTextAlign = 'left' | 'center' | 'right';
export type RichTextHref = ( () => void ) | string;
type RichTextLinksObject = Record<string, RichTextHref>;
export type RichTextLinks = RichTextLinksObject | true;

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

  // Sets the map of href placeholder => actual href/callback used for links. However, if set to true ({boolean}) as a
  // full object, links in the string will not be mapped, but will be directly added.
  //
  // For instance, the default is to map hrefs for security purposes:
  //
  // new RichText( '<a href="{{alink}}">content</a>', {
  //   links: {
  //     alink: 'https://phet.colorado.edu'
  //   }
  // } );
  //
  // But links with an href not matching will be ignored. This can be avoided by passing links: true to directly
  // embed links:
  //
  // new RichText( '<a href="https://phet.colorado.edu">content</a>', { links: true } );
  //
  // Callbacks (instead of a URL) are also supported, e.g.:
  //
  // new RichText( '<a href="{{acallback}}">content</a>', {
  //   links: {
  //     acallback: function() { console.log( 'clicked' ) }
  //   }
  // } );
  //
  // See https://github.com/phetsims/scenery-phet/issues/316 for more information.
  links?: RichTextLinks;

  // A map of string => Node, where `<node id="string"/>` will get replaced by the given Node (DAG supported)
  //
  // For example:
  //
  // new RichText( 'This is a <node id="test"/>', {
  //   nodes: {
  //     test: new Text( 'Node' )
  //   }
  // }
  //
  // Alignment is also supported, with the align attribute (center/top/bottom/origin).
  // This alignment is in relation to the current text/font size in the HTML where the <node> tag is placed.
  // An example:
  //
  // new RichText( 'This is a <node id="test" align="top"/>', {
  //   nodes: {
  //     test: new Text( 'Node' )
  //   }
  // }
  // NOTE: When alignment isn't supplied, origin is used as a default. Origin means "y=0 is placed at the baseline of
  // the text".
  nodes?: Record<string, Node>;

  // Will replace newlines (`\n`) with <br>, similar to the old MultiLineText (defaults to false)
  replaceNewlines?: boolean;

  // Sets text alignment if there are multiple lines
  align?: RichTextAlign;

  // Sets the spacing between lines if there are multiple lines
  leading?: number;

  // Sets width of text before creating a new line
  lineWrap?: number | null;

  // Sets forwarding of the stringProperty, see setStringProperty() for more documentation
  stringProperty?: TReadOnlyProperty<string> | null;

  stringPropertyOptions?: PropertyOptions<string>;

  // Sets the string to be displayed by this Node
  string?: string | number;
};

export type RichTextOptions = SelfOptions & NodeOptions;

const DEFAULT_FONT = new Font( {
  size: 20
} );

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

// We'll store an array here that will record which links/nodes were used in the last rebuild (so we can assert out if
// there were some that were not used).
const usedLinks: string[] = [];
const usedNodes: string[] = [];

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

const FONT_STYLE_KEYS = Object.keys( FONT_STYLE_MAP ) as ( keyof typeof FONT_STYLE_MAP )[];
const STYLE_KEYS = [ 'color' ].concat( FONT_STYLE_KEYS );

export default class RichText extends Node {

  // The string to display. We'll initialize this by mutating.
  private readonly _stringProperty: TinyForwardingProperty<string>;

  private _font: Font | string = DEFAULT_FONT;
  private _boundsMethod: TextBoundsMethod = 'hybrid';
  private _fill: TPaint = '#000000';
  private _stroke: TPaint = null;
  private _lineWidth = 1;

  private _subScale = 0.75;
  private _subXSpacing = 0;
  private _subYOffset = 0;

  private _supScale = 0.75;
  private _supXSpacing = 0;
  private _supYOffset = 0;

  private _capHeightScale = 0.75;

  private _underlineLineWidth = 1;
  private _underlineHeightScale = 0.15;

  private _strikethroughLineWidth = 1;
  private _strikethroughHeightScale = 0.3;

  private _linkFill: TPaint = 'rgb(27,0,241)';

  private _linkEventsHandled = false;

  // If an object, values are either {string} or {function}
  private _links: RichTextLinks = {};

  private _nodes: Record<string, Node> = {};

  private _replaceNewlines = false;
  private _align: RichTextAlign = 'left';
  private _leading = 0;
  private _lineWrap: number | null = null;

  // We need to consolidate links (that could be split across multiple lines) under one "link" node, so we track created
  // link fragments here so they can get pieced together later.
  private _linkItems: { element: HimalayaNode; node: Node; href: RichTextHref }[] = [];

  // Whether something has been added to this line yet. We don't want to infinite-loop out if something is longer than
  // our lineWrap, so we'll place one item on its own on an otherwise empty line.
  private _hasAddedLeafToLine = false;

  // Normal layout container of lines (separate, so we can clear it easily)
  private lineContainer: Node;

  // Text and RichText currently use the same tandem name for their stringProperty.
  public static readonly STRING_PROPERTY_TANDEM_NAME = Text.STRING_PROPERTY_TANDEM_NAME;

  public constructor( string: string | number | TReadOnlyProperty<string>, providedOptions?: RichTextOptions ) {

    // We only fill in some defaults, since the other defaults are defined below (and mutate is relied on)
    const options = optionize<RichTextOptions, Pick<SelfOptions, 'fill'>, NodeOptions>()( {
      fill: '#000000',

      // phet-io
      tandem: Tandem.OPTIONAL,
      tandemNameSuffix: 'Text',
      phetioType: RichText.RichTextIO,
      phetioVisiblePropertyInstrumented: false
    }, providedOptions );

    if ( typeof string === 'string' || typeof string === 'number' ) {
      options.string = string;
    }
    else {
      options.stringProperty = string;
    }

    super();

    this._stringProperty = new TinyForwardingProperty( '', true, this.onStringPropertyChange.bind( this ) );

    this.lineContainer = new Node( {} );
    this.addChild( this.lineContainer );

    // Initialize to an empty state, so we are immediately valid (since now we need to create an empty leaf even if we
    // have empty text).
    this.rebuildRichText();

    this.mutate( options );
  }

  /**
   * Called when our stringProperty changes values.
   */
  private onStringPropertyChange(): void {
    this.rebuildRichText();
  }

  /**
   * See documentation for Node.setVisibleProperty, except this is for the text string.
   */
  public setStringProperty( newTarget: TProperty<string> | null ): this {
    return this._stringProperty.setTargetProperty( this, RichText.STRING_PROPERTY_TANDEM_NAME, newTarget );
  }

  public set stringProperty( property: TProperty<string> | null ) { this.setStringProperty( property ); }

  public get stringProperty(): TProperty<string> { return this.getStringProperty(); }

  /**
   * Like Node.getVisibleProperty, but for the text string. Note this is not the same as the Property provided in
   * setStringProperty. Thus is the nature of TinyForwardingProperty.
   */
  public getStringProperty(): TProperty<string> {
    return this._stringProperty;
  }

  /**
   * See documentation and comments in Node.initializePhetioObject
   */
  public override initializePhetioObject( baseOptions: Partial<PhetioObjectOptions>, providedOptions: RichTextOptions ): void {

    const options = optionize<RichTextOptions, EmptySelfOptions, RichTextOptions>()( {}, providedOptions );

    // Track this, so we only override our stringProperty once.
    const wasInstrumented = this.isPhetioInstrumented();

    super.initializePhetioObject( baseOptions, options );

    if ( Tandem.PHET_IO_ENABLED && !wasInstrumented && this.isPhetioInstrumented() ) {

      this._stringProperty.initializePhetio( this, RichText.STRING_PROPERTY_TANDEM_NAME, () => {
        return new StringProperty( this.string, combineOptions<RichTextOptions>( {

          // by default, texts should be readonly. Editable texts most likely pass in editable Properties from i18n model Properties, see https://github.com/phetsims/scenery/issues/1443
          phetioReadOnly: true,
          tandem: this.tandem.createTandem( RichText.STRING_PROPERTY_TANDEM_NAME ),
          phetioDocumentation: 'Property for the displayed text'
        }, options.stringPropertyOptions ) );
      } );
    }
  }

  /**
   * When called, will rebuild the node structure for this RichText
   */
  private rebuildRichText(): void {
    assert && cleanArray( usedLinks );
    assert && cleanArray( usedNodes );

    this.freeChildrenToPool();

    // Bail early, particularly if we are being constructed.
    if ( this.string === '' ) {
      this.appendEmptyLeaf();
      return;
    }

    sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `RichText#${this.id} rebuild` );
    sceneryLog && sceneryLog.RichText && sceneryLog.push();

    // Turn bidirectional marks into explicit elements, so that the nesting is applied correctly.
    let mappedText = this.string.replace( /\u202a/g, '<span dir="ltr">' )
      .replace( /\u202b/g, '<span dir="rtl">' )
      .replace( /\u202c/g, '</span>' );

    // Optional replacement of newlines, see https://github.com/phetsims/scenery/issues/1542
    if ( this._replaceNewlines ) {
      mappedText = mappedText.replace( /\n/g, '<br>' );
    }

    let rootElements: HimalayaNode[];

    // Start appending all top-level elements
    try {
      // @ts-expect-error - Since himalaya isn't in tsconfig
      rootElements = himalaya.parse( mappedText );
    }
    catch( e ) {
      // If we error out, don't kill the sim. Instead, replace the string with something that looks obviously like an
      // error. See https://github.com/phetsims/chipper/issues/1361 (we don't want translations to error out our
      // build process).

      // @ts-expect-error - Since himalaya isn't in tsconfig
      rootElements = himalaya.parse( 'INVALID TRANSLATION' );
    }

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
          this.appendLine( RichTextVerticalSpacer.pool.create( RichTextUtils.scratchText.setString( 'X' ).setFont( this._font ).height ) );
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

    if ( assert ) {
      if ( this._links && this._links !== true ) {
        Object.keys( this._links ).forEach( link => {
          assert && allowLinksProperty.value && assert( usedLinks.includes( link ), `Unused RichText link: ${link}` );
        } );
      }
      if ( this._nodes ) {
        Object.keys( this._nodes ).forEach( node => {
          assert && allowLinksProperty.value && assert( usedNodes.includes( node ), `Unused RichText node: ${node}` );
        } );
      }
    }

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

      child.clean();
    }
  }

  /**
   * Releases references.
   */
  public override dispose(): void {
    this.freeChildrenToPool();

    super.dispose();

    this._stringProperty.dispose();
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
   * @param isLTR - True if LTR, false if RTL (handles RTL strings properly)
   * @param widthAvailable - How much width we have available before forcing a line break (for lineWrap)
   * @returns - Whether a line break was reached
   */
  private appendElement( containerNode: RichTextElement, element: HimalayaNode, font: Font | string, fill: TPaint, isLTR: boolean, widthAvailable: number ): string {
    let lineBreakState = LineBreakState.NONE;

    // The main Node for the element that we are adding
    let node!: RichTextLeaf | RichTextNode | RichTextElement;

    // If this content gets added, it will need to be pushed over by this amount
    const containerSpacing = isLTR ? containerNode.rightSpacing : containerNode.leftSpacing;

    // Container spacing cuts into our effective available width
    const widthAvailableWithSpacing = widthAvailable - containerSpacing;

    // If we're a leaf
    if ( isHimalayaTextNode( element ) ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `appending leaf: ${element.content}` );
      sceneryLog && sceneryLog.RichText && sceneryLog.push();

      node = RichTextLeaf.pool.create( element.content, isLTR, font, this._boundsMethod, fill, this._stroke, this._lineWidth );

      // Handle wrapping if required. Container spacing cuts into our available width
      if ( !node.fitsIn( widthAvailableWithSpacing, this._hasAddedLeafToLine, isLTR ) ) {
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
            node.clean(); // We're tossing the old one, so we'll free up memory for the new one
            node = RichTextLeaf.pool.create( rangesToString( ranges ), isLTR, font, this._boundsMethod, fill, this._stroke, this._lineWidth );

            // If we haven't added anything to the line AND we are down to the first word, we need to just add it.
            if ( !node.fitsIn( widthAvailableWithSpacing, this._hasAddedLeafToLine, isLTR ) &&
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
            // We won't use this one, so we'll free it back to the pool
            node.clean();

            return LineBreakState.INCOMPLETE;
          }
        }
      }

      this._hasAddedLeafToLine = true;

      sceneryLog && sceneryLog.RichText && sceneryLog.pop();
    }
    // Otherwise presumably an element with content
    else if ( isHimalayaElementNode( element ) ) {
      // Bail out quickly for a line break
      if ( element.tagName === 'br' ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'manual line break' );
        return LineBreakState.COMPLETE;
      }

      // Span (dir attribute) -- we need the LTR/RTL knowledge before most other operations
      if ( element.tagName === 'span' ) {
        const dirAttributeString = RichTextUtils.himalayaGetAttribute( 'dir', element );

        if ( dirAttributeString ) {
          assert && assert( dirAttributeString === 'ltr' || dirAttributeString === 'rtl',
            'Span dir attributes should be ltr or rtl.' );
          isLTR = dirAttributeString === 'ltr';
        }
      }

      // Handle <node> tags, which should not have content
      if ( element.tagName === 'node' ) {
        const referencedId = RichTextUtils.himalayaGetAttribute( 'id', element );
        const referencedNode = referencedId ? ( this._nodes[ referencedId ] || null ) : null;

        assert && assert( referencedNode,
          referencedId
          ? `Could not find a matching item in RichText's nodes for ${referencedId}. It should be provided in the nodes option`
          : 'No id attribute provided for a given <node> element' );
        if ( referencedNode ) {
          assert && usedNodes.push( referencedId! );
          node = RichTextNode.pool.create( referencedNode );

          if ( this._hasAddedLeafToLine && !node.fitsIn( widthAvailableWithSpacing ) ) {
            // If we don't fit, we'll toss this node to the pool and create it on the next line
            node.clean();
            return LineBreakState.INCOMPLETE;
          }

          const nodeAlign = RichTextUtils.himalayaGetAttribute( 'align', element );
          if ( nodeAlign === 'center' || nodeAlign === 'top' || nodeAlign === 'bottom' ) {
            const textBounds = RichTextUtils.scratchText.setString( 'Test' ).setFont( font ).bounds;
            if ( nodeAlign === 'center' ) {
              node.centerY = textBounds.centerY;
            }
            else if ( nodeAlign === 'top' ) {
              node.top = textBounds.top;
            }
            else if ( nodeAlign === 'bottom' ) {
              node.bottom = textBounds.bottom;
            }
          }
        }
        else {
          // If there is no node in our map, we'll just skip it
          return lineBreakState;
        }

        this._hasAddedLeafToLine = true;
      }
      // If not a <node> tag
      else {
        node = RichTextElement.pool.create( isLTR );
      }

      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'appending element' );
      sceneryLog && sceneryLog.RichText && sceneryLog.push();

      const styleAttributeString = RichTextUtils.himalayaGetAttribute( 'style', element );

      if ( styleAttributeString ) {
        const css = RichTextUtils.himalayaStyleStringToMap( styleAttributeString );
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
            fontOptions[ FONT_STYLE_MAP[ styleKey ] ] = css[ styleKey ];
          }
        }
        font = ( typeof font === 'string' ? Font.fromCSS( font ) : font ).copy( fontOptions );
      }

      // Anchor (link)
      if ( element.tagName === 'a' ) {
        let href: RichTextHref | null = RichTextUtils.himalayaGetAttribute( 'href', element );
        const originalHref = href;

        // Try extracting the href from the links object
        if ( href !== null && this._links !== true ) {
          if ( href.startsWith( '{{' ) && href.indexOf( '}}' ) === href.length - 2 ) {
            const linkName = href.slice( 2, -2 );
            href = this._links[ linkName ];
            assert && usedLinks.push( linkName );
          }
          else {
            href = null;
          }
        }

        // Ignore things if there is no matching href
        assert && assert( href,
          `Could not find a matching item in RichText's links for ${originalHref}. It should be provided in the links option, or links should be turned to true (to allow the string to create its own urls` );
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
      if ( element.tagName !== 'node' ) {
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
          node.centerY = RichTextUtils.scratchText.setString( 'X' ).setFont( font ).top * this._capHeightScale;
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
      const wasAdded = containerNode.addElement( node );
      if ( !wasAdded ) {
        // Remove it from the linkItems if we didn't actually add it.
        this._linkItems = this._linkItems.filter( item => item.node !== node );

        // And since we won't dispose it (since it's not a child), clean it here
        node.clean();
      }
    }

    return lineBreakState;
  }

  /**
   * Sets the string displayed by our node.
   *
   * NOTE: Encoding HTML entities is required, and malformed HTML is not accepted.
   *
   * @param string - The string to display. If it's a number, it will be cast to a string
   */
  public setString( string: string | number ): this {
    assert && assert( string !== null && string !== undefined, 'String should be defined and non-null. Use the empty string if needed.' );

    // cast it to a string (for numbers, etc., and do it before the change guard so we don't accidentally trigger on non-changed string)
    string = `${string}`;

    this._stringProperty.set( string );

    return this;
  }

  public set string( value: string | number ) { this.setString( value ); }

  public get string(): string { return this.getString(); }

  /**
   * Returns the string displayed by our text Node.
   */
  public getString(): string {
    return this._stringProperty.value;
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
   * Sets the scale (relative to 1) of any string under subscript (<sub>) elements.
   */
  public setSubScale( subScale: number ): this {
    assert && assert( isFinite( subScale ) && subScale > 0 );

    if ( this._subScale !== subScale ) {
      this._subScale = subScale;
      this.rebuildRichText();
    }
    return this;
  }

  public set subScale( value: number ) { this.setSubScale( value ); }

  public get subScale(): number { return this.getSubScale(); }

  /**
   * Returns the scale (relative to 1) of any string under subscript (<sub>) elements.
   */
  public getSubScale(): number {
    return this._subScale;
  }

  /**
   * Sets the horizontal spacing before any subscript (<sub>) elements.
   */
  public setSubXSpacing( subXSpacing: number ): this {
    assert && assert( isFinite( subXSpacing ) );

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
    assert && assert( isFinite( subYOffset ) );

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
   * Sets the scale (relative to 1) of any string under superscript (<sup>) elements.
   */
  public setSupScale( supScale: number ): this {
    assert && assert( isFinite( supScale ) && supScale > 0 );

    if ( this._supScale !== supScale ) {
      this._supScale = supScale;
      this.rebuildRichText();
    }
    return this;
  }

  public set supScale( value: number ) { this.setSupScale( value ); }

  public get supScale(): number { return this.getSupScale(); }

  /**
   * Returns the scale (relative to 1) of any string under superscript (<sup>) elements.
   */
  public getSupScale(): number {
    return this._supScale;
  }

  /**
   * Sets the horizontal spacing before any superscript (<sup>) elements.
   */
  public setSupXSpacing( supXSpacing: number ): this {
    assert && assert( isFinite( supXSpacing ) );

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
    assert && assert( isFinite( supYOffset ) );

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
    assert && assert( isFinite( capHeightScale ) && capHeightScale > 0 );

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
    assert && assert( isFinite( underlineLineWidth ) && underlineLineWidth > 0 );

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
    assert && assert( isFinite( underlineHeightScale ) && underlineHeightScale > 0 );

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
    assert && assert( isFinite( strikethroughLineWidth ) && strikethroughLineWidth > 0 );

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
    assert && assert( isFinite( strikethroughHeightScale ) && strikethroughHeightScale > 0 );

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

  public setLinks( links: RichTextLinks ): this {
    assert && assert( links === true || Object.getPrototypeOf( links ) === Object.prototype );

    if ( this._links !== links ) {
      this._links = links;
      this.rebuildRichText();
    }
    return this;
  }

  /**
   * Returns whether link events will be handled.
   */
  public getLinks(): RichTextLinks {
    return this._links;
  }

  public set links( value: RichTextLinks ) { this.setLinks( value ); }

  public get links(): RichTextLinks { return this.getLinks(); }

  public setNodes( nodes: Record<string, Node> ): this {
    assert && assert( Object.getPrototypeOf( nodes ) === Object.prototype );

    if ( this._nodes !== nodes ) {
      this._nodes = nodes;
      this.rebuildRichText();
    }

    return this;
  }

  public getNodes(): Record<string, Node> {
    return this._nodes;
  }

  public set nodes( value: Record<string, Node> ) { this.setNodes( value ); }

  public get nodes(): Record<string, Node> { return this.getNodes(); }

  /**
   * Sets whether newlines are replaced with <br>
   */
  public setReplaceNewlines( replaceNewlines: boolean ): this {
    if ( this._replaceNewlines !== replaceNewlines ) {
      this._replaceNewlines = replaceNewlines;
      this.rebuildRichText();
    }
    return this;
  }

  public set replaceNewlines( value: boolean ) { this.setReplaceNewlines( value ); }

  public get replaceNewlines(): boolean { return this.getReplaceNewlines(); }

  public getReplaceNewlines(): boolean {
    return this._replaceNewlines;
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
    assert && assert( isFinite( leading ) );

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
    assert && assert( lineWrap === null || ( isFinite( lineWrap ) && lineWrap > 0 ) );

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

    if ( assert && options && options.hasOwnProperty( 'string' ) && options.hasOwnProperty( Text.STRING_PROPERTY_NAME ) && options.stringProperty ) {
      assert && assert( options.stringProperty.value === options.string, 'If both string and stringProperty are provided, then values should match' );
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
    if ( isHimalayaTextNode( element ) ) {
      return RichText.contentToString( element.content, isLTR );
    }
    else if ( isHimalayaElementNode( element ) ) {
      const dirAttributeString = RichTextUtils.himalayaGetAttribute( 'dir', element );

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
    if ( isHimalayaTextNode( element ) ) {
      return RichText.contentToString( element.content, isLTR );
    }
    else if ( isHimalayaElementNode( element ) ) {
      const dirAttribute = RichTextUtils.himalayaGetAttribute( 'dir', element );

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
    // @ts-expect-error - we should get a string from this
    const unescapedContent: string = he.decode( content );
    return isLTR ? ( `\u202a${unescapedContent}\u202c` ) : ( `\u202b${unescapedContent}\u202c` );
  }

  public static RichTextIO: IOType;
}

/**
 * {Array.<string>} - String keys for all the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
RichText.prototype._mutatorKeys = RICH_TEXT_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'RichText', RichText );

RichText.RichTextIO = new IOType( 'RichTextIO', {
  valueType: RichText,
  supertype: Node.NodeIO,
  documentation: 'The tandem IO Type for the scenery RichText node'
} );
