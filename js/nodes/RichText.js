// Copyright 2017-2021, University of Colorado Boulder

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

import StringProperty from '../../../axon/js/StringProperty.js';
import TinyForwardingProperty from '../../../axon/js/TinyForwardingProperty.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Poolable from '../../../phet-core/js/Poolable.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import memoize from '../../../phet-core/js/memoize.js';
import merge from '../../../phet-core/js/merge.js';
import openPopup from '../../../phet-core/js/openPopup.js';
import Tandem from '../../../tandem/js/Tandem.js';
import IOType from '../../../tandem/js/types/IOType.js';
import FireListener from '../listeners/FireListener.js';
import scenery from '../scenery.js';
import Color from '../util/Color.js';
import Font from '../util/Font.js';
import Line from './Line.js';
import Node from './Node.js';
import Text from './Text.js';
import VStrut from './VStrut.js';

// Options that can be used in the constructor, with mutate(), or directly as setters/getters
// each of these options has an associated setter, see setter methods for more documentation
const RICH_TEXT_OPTION_KEYS = [
  'boundsMethod', // {string} - Sets how bounds are determined for text
  'font', // {Font|string} - Sets the font for the text
  'fill', // {PaintDef} - Sets the fill of the text
  'stroke', // {PaintDef} - Sets the stroke around the text
  'lineWidth', // {number} - Sets the lineWidth around the text
  'subScale', // {number} - Sets the scale of any subscript elements
  'subXSpacing', // {number} - Sets horizontal spacing before any subscript elements
  'subYOffset', // {number} - Sets vertical offset for any subscript elements
  'supScale', // {number} - Sets the scale for any superscript elements
  'supXSpacing', // {number} - Sets the horizontal offset before any superscript elements
  'supYOffset', // {number} - Sets the vertical offset for any superscript elements
  'capHeightScale', // {number} - Sets the expected cap height cap height (baseline to top of capital letters) as a scale
  'underlineLineWidth', // {number} - Sets the line width for underlines
  'underlineHeightScale', // {number} - Sets the underline height as a scale relative to text bounds height
  'strikethroughLineWidth', // {number} - Sets line width for strikethrough
  'strikethroughHeightScale', // {number} - Sets height of strikethrough as a scale relative to text bounds height
  'linkFill', // {paint} - Sets the fill for links within the text
  'linkEventsHandled', // {boolean} - Sets whether link clicks will call event.handle()
  'links', // {Object|boolean} - Sets the map of href placeholder => actual href/callback used for links
  'align', // {string} - Sets text alignment if there are multiple lines
  'leading', // {number} - Sets the spacing between lines if there are multiple lines
  'lineWrap', // {number|null} - Sets width of text before creating a new line
  'textProperty', // {Property.<string>|null} - Sets forwarding of the textProperty, see setTextProperty() for more documentation
  'text' // {string|number} - Sets the text to be displayed by this Node
];

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

// We need to do some font-size tests, so we have a Text for that.
const scratchText = new Text( '' );

// himalaya converts dash separated CSS to camel case - use CSS compatible style with dashes, see above for examples
const FONT_STYLE_MAP = {
  fontFamily: 'family',
  fontSize: 'size',
  fontStretch: 'stretch',
  fontStyle: 'style',
  fontVariant: 'variant',
  fontWeight: 'weight',
  lineHeight: 'lineHeight'
};

const FONT_STYLE_KEYS = Object.keys( FONT_STYLE_MAP );
const STYLE_KEYS = [ 'color' ].concat( FONT_STYLE_KEYS );

class RichText extends Node {
  /**
   * @public
   *
   * @param {string|number} text
   * @param {Object} [options] - RichText-specific options are documented in RICH_TEXT_OPTION_KEYS above, and can be
   *                             provided along-side options for Node.
   */
  constructor( text, options ) {

    super();

    // @public {TinyProperty.<string>} - The text to display. We'll initialize this by mutating.
    this._textProperty = new TinyForwardingProperty( '', true, this.onTextPropertyChange.bind( this ) );

    // @private {Font}
    this._font = DEFAULT_FONT;

    // @private {string}
    this._boundsMethod = 'hybrid';

    // @private {PaintDef}
    this._fill = '#000000';
    this._stroke = null;

    // @private {number}
    this._lineWidth = 1;

    // @private {number}
    this._subScale = 0.75;
    this._subXSpacing = 0;
    this._subYOffset = 0;

    // @private {number}
    this._supScale = 0.75;
    this._supXSpacing = 0;
    this._supYOffset = 0;

    // @private {number}
    this._capHeightScale = 0.75;

    // @private {number}
    this._underlineLineWidth = 1;
    this._underlineHeightScale = 0.15;

    // @private {number}
    this._strikethroughLineWidth = 1;
    this._strikethroughHeightScale = 0.3;

    // @private {paint}
    this._linkFill = 'rgb(27,0,241)';

    // @private {boolean}
    this._linkEventsHandled = false;

    // @private {Object|boolean} - If an object, values are either {string} or {function}
    this._links = {};

    // @private {string}
    this._align = 'left'; // 'left', 'center', or 'right'

    // @private {number}
    this._leading = 0;

    // @private {number|null}
    this._lineWrap = null;

    // @private {Array.<{ element: {*}, node: {Node}, href: {string} }>} - We need to consolidate links (that could be
    // split across multiple lines) under one "link" node, so we track created link fragments here so they can get
    // pieced together later.
    this._linkItems = [];

    // @private {boolean} - Whether something has been added to this line yet. We don't want to infinite-loop out if
    // something is longer than our lineWrap, so we'll place one item on its own on an otherwise empty line.
    this._hasAddedLeafToLine = false;

    options = extendDefined( {
      fill: '#000000',
      text: text,
      tandem: Tandem.OPTIONAL,
      phetioType: RichText.RichTextIO
    }, options );

    // @private {Node} - Normal layout container of lines (separate, so we can clear it easily)
    this.lineContainer = new Node( {} );
    this.addChild( this.lineContainer );

    // Initialize to an empty state, so we are immediately valid (since now we need to create an empty leaf even if we
    // have empty text).
    this.rebuildRichText();

    this.mutate( options );
  }


  /**
   * Called when our text Property changes values.
   * @private
   */
  onTextPropertyChange() {
    this.rebuildRichText();
  }

  /**
   * See documentation for Node.setVisibleProperty, except this is for the text string.
   *
   * @public
   *
   * @param {TinyProperty.<string>|Property.<string>|null} newTarget
   * @returns {Text} for chaining
   */
  setTextProperty( newTarget ) {
    return this._textProperty.setTargetProperty( this, TEXT_PROPERTY_TANDEM_NAME, newTarget );
  }

  set textProperty( property ) { this.setTextProperty( property ); }

  /**
   * Like Node.getVisibleProperty, but for the text string. Note this is not the same as the Property provided in
   * setTextProperty. Thus is the nature of TinyForwardingProperty.
   *
   * @returns {TinyForwardingProperty}
   * @public
   */
  getTextProperty() {
    return this._textProperty;
  }

  get textProperty() { return this.getTextProperty(); }

  /**
   * See documentation and comments in Node.initializePhetioObject
   * @param {Object} baseOptions
   * @param {Object} config
   * @override
   * @protected
   */
  initializePhetioObject( baseOptions, config ) {

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
        }, config.textPropertyOptions ) )
      );
    }
  }

  /**
   * When called, will rebuild the node structure for this RichText
   * @private
   */
  rebuildRichText() {
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
    const rootElements = himalaya.parse( mappedText );

    // Clear out link items, as we'll need to reconstruct them later
    this._linkItems.length = 0;

    const widthAvailable = this._lineWrap === null ? Number.POSITIVE_INFINITY : this._lineWrap;
    const isRootLTR = true;

    let currentLine = RichTextElement.createFromPool( isRootLTR );
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
        currentLine = RichTextElement.createFromPool( isRootLTR );
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

        const linkRootNode = RichTextLink.createFromPool( linkElement.innerContent, href );
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
   * @private
   */
  freeChildrenToPool() {
    // Clear any existing lines or link fragments (higher performance, and return them to pools also)
    while ( this.lineContainer._children.length ) {
      const child = this.lineContainer._children[ this.lineContainer._children.length - 1 ];
      this.lineContainer.removeChild( child );
      child.clean();
    }
  }

  /**
   * Releases references.
   * @public
   * @override
   */
  dispose() {
    this.freeChildrenToPool();

    super.dispose();

    this._textProperty.dispose();
  }

  /**
   * Appends a finished line, applying any necessary leading.
   * @private
   *
   * @param {RichTextElement} lineNode
   */
  appendLine( lineNode ) {
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
   * @private
   */
  appendEmptyLeaf() {
    assert && assert( this.lineContainer.getChildrenCount() === 0 );

    this.appendLine( RichTextLeaf.createFromPool( '', true, this._font, this._boundsMethod, this._fill, this._stroke, this._lineWidth ) );
  }

  /**
   * Aligns all lines attached to the lineContainer.
   * @private
   */
  alignLines() {
    // All nodes will either share a 'left', 'centerX' or 'right'.
    const coordinateName = this._align === 'center' ? 'centerX' : this._align;

    const ideal = this.lineContainer[ coordinateName ];
    for ( let i = 0; i < this.lineContainer.getChildrenCount(); i++ ) {
      this.lineContainer.getChildAt( i )[ coordinateName ] = ideal;
    }
  }

  /**
   * Main recursive function for constructing the RichText Node tree.
   * @private
   *
   * We'll add any relevant content to the containerNode. The element will be mutated as things are added, so that
   * whenever content is added to the Node tree it will be removed from the element tree. This means we can pause
   * whenever (e.g. when a line-break is encountered) and the rest will be ready for parsing the next line.
   *
   * @param {RichTextElement} containerNode - The node where child elements should be placed
   * @param {*} element - See Himalaya's element specification
   *                      (https://github.com/andrejewski/himalaya/blob/master/text/ast-spec-v0.md)
   * @param {Font|string} font - The font to apply at this level
   * @param {PaintDef} fill - Fill to apply
   * @param {boolean} isLTR - True if LTR, false if RTL (handles RTL text properly)
   * @param {number} widthAvailable - How much width we have available before forcing a line break (for lineWrap)
   * @returns {LineBreakState} - Whether a line break was reached
   */
  appendElement( containerNode, element, font, fill, isLTR, widthAvailable ) {
    let lineBreakState = LineBreakState.NONE;

    // {Node|Text} - The main Node for the element that we are adding
    let node;

    // If we're a leaf
    if ( element.type === 'Text' ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `appending leaf: ${element.content}` );
      sceneryLog && sceneryLog.RichText && sceneryLog.push();

      node = RichTextLeaf.createFromPool( element.content, isLTR, font, this._boundsMethod, fill, this._stroke, this._lineWidth );

      // If this content gets added, it will need to be pushed over by this amount
      const containerSpacing = isLTR ? containerNode.rightSpacing : containerNode.leftSpacing;

      // Handle wrapping if required. Container spacing cuts into our available width
      if ( !node.fitsIn( widthAvailable - containerSpacing, this._hasAddedLeafToLine, isLTR ) ) {
        // Didn't fit, lets break into words to see what we can fit
        const words = element.content.split( ' ' );

        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `Overflow leafAdded:${this._hasAddedLeafToLine}, words: ${words.length}` );

        // If we need to add something (and there is only a single word), then add it
        if ( this._hasAddedLeafToLine || words.length > 1 ) {
          sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Skipping words' );

          const skippedWords = [];
          let success = false;
          skippedWords.unshift( words.pop() ); // We didn't fit with the last one!

          // Keep shortening by removing words until it fits (or if we NEED to fit it) or it doesn't fit.
          while ( words.length ) {
            node = RichTextLeaf.createFromPool( words.join( ' ' ), isLTR, font, this._boundsMethod, fill, this._stroke, this._lineWidth );

            // If we haven't added anything to the line and we are down to the first word, we need to just add it.
            if ( !node.fitsIn( widthAvailable - containerSpacing, this._hasAddedLeafToLine, isLTR ) &&
                 ( this._hasAddedLeafToLine || words.length > 1 ) ) {
              sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `Skipping word ${words[ words.length - 1 ]}` );
              skippedWords.unshift( words.pop() );
            }
            else {
              sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `Success with ${words.join( ' ' )}` );
              success = true;
              break;
            }
          }

          // If we haven't added anything yet to this line, we'll permit the overflow
          if ( success ) {
            lineBreakState = LineBreakState.INCOMPLETE;
            element.content = skippedWords.join( ' ' );
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
    else if ( element.type === 'Element' ) {
      // Bail out quickly for a line break
      if ( element.tagName === 'br' ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'manual line break' );
        return LineBreakState.COMPLETE;
      }
      // Span (dir attribute) -- we need the LTR/RTL knowledge before most other operations
      else if ( element.tagName === 'span' ) {
        if ( element.attributes.dir ) {
          assert && assert( element.attributes.dir === 'ltr' || element.attributes.dir === 'rtl',
            'Span dir attributes should be ltr or rtl.' );
          isLTR = element.attributes.dir === 'ltr';
        }
      }

      node = RichTextElement.createFromPool( isLTR );

      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'appending element' );
      sceneryLog && sceneryLog.RichText && sceneryLog.push();

      if ( element.attributes.style ) {
        const css = element.attributes.style;
        assert && Object.keys( css ).forEach( key => {
          assert( _.includes( STYLE_KEYS, key ), 'See supported style CSS keys' );
        } );

        // Fill
        if ( css.color ) {
          fill = new Color( css.color );
        }

        // Font
        const fontOptions = {};
        for ( let i = 0; i < FONT_STYLE_KEYS.length; i++ ) {
          const styleKey = FONT_STYLE_KEYS[ i ];
          if ( css[ styleKey ] ) {
            fontOptions[ FONT_STYLE_MAP[ styleKey ] ] = css[ styleKey ];
          }
        }
        font = font.copy( fontOptions );
      }

      // Achor (link)
      if ( element.tagName === 'a' ) {
        let href = element.attributes.href;

        // Try extracting the href from the links object
        if ( this._links !== true ) {
          if ( href.indexOf( '{{' ) === 0 && href.indexOf( '}}' ) === href.length - 2 ) {
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
        font = font.copy( {
          weight: 'bold'
        } );
      }
      // Italic
      else if ( element.tagName === 'i' || element.tagName === 'em' ) {
        font = font.copy( {
          style: 'italic'
        } );
      }
      // Subscript
      else if ( element.tagName === 'sub' ) {
        node.scale( this._subScale );
        node.addExtraBeforeSpacing( this._subXSpacing );
        node.y += this._subYOffset;
      }
      // Superscript
      else if ( element.tagName === 'sup' ) {
        node.scale( this._supScale );
        node.addExtraBeforeSpacing( this._supXSpacing );
        node.y += this._supYOffset;
      }

      // If we've added extra spacing, we'll need to subtract it off of our available width
      const scale = node.getScaleVector().x;

      // Process children
      while ( lineBreakState === LineBreakState.NONE && element.children.length ) {
        const widthBefore = node.bounds.isValid() ? node.width : 0;

        const childElement = element.children[ 0 ];
        lineBreakState = this.appendElement( node, childElement, font, fill, isLTR, widthAvailable / scale );

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
          node.addChild( new Line( node.localBounds.left, underlineY, node.localBounds.right, underlineY, {
            stroke: fill,
            lineWidth: this._underlineLineWidth
          } ) );
        }
      }
      // Strikethrough
      else if ( element.tagName === 's' ) {
        const strikethroughY = node.top * this._strikethroughHeightScale;
        if ( isFinite( node.top ) ) {
          node.addChild( new Line( node.localBounds.left, strikethroughY, node.localBounds.right, strikethroughY, {
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
   * Sets the text displayed by our node.
   * @public
   *
   * NOTE: Encoding HTML entities is required, and malformed HTML is not accepted.
   *
   * @param {string|number} text - The text to display. If it's a number, it will be cast to a string
   * @returns {RichText} - For chaining
   */
  setText( text ) {
    assert && assert( text !== null && text !== undefined, 'Text should be defined and non-null. Use the empty string if needed.' );
    assert && assert( typeof text === 'number' || typeof text === 'string', 'text should be a string or number' );

    // cast it to a string (for numbers, etc., and do it before the change guard so we don't accidentally trigger on non-changed text)
    text = `${text}`;

    this._textProperty.set( text );

    return this;
  }

  set text( value ) { this.setText( value ); }

  /**
   * Returns the text displayed by our node.
   * @public
   *
   * @returns {string}
   */
  getText() {
    return this._textProperty.value;
  }

  get text() { return this.getText(); }

  /**
   * Sets the method that is used to determine bounds from the text. See Text.setBoundsMethod for details
   * @public
   *
   * @param {string} method
   * @returns {RichText} - For chaining.
   */
  setBoundsMethod( method ) {
    assert && assert( method === 'fast' || method === 'fastCanvas' || method === 'accurate' || method === 'hybrid', 'Unknown Text boundsMethod' );
    if ( method !== this._boundsMethod ) {
      this._boundsMethod = method;
      this.rebuildRichText();
    }
    return this;
  }

  set boundsMethod( value ) { this.setBoundsMethod( value ); }

  /**
   * Returns the current method to estimate the bounds of the text. See setBoundsMethod() for more information.
   * @public
   *
   * @returns {string}
   */
  getBoundsMethod() {
    return this._boundsMethod;
  }

  get boundsMethod() { return this.getBoundsMethod(); }

  /**
   * Sets the font of our node.
   * @public
   *
   * @param {Font|string} font
   * @returns {RichText} - For chaining.
   */
  setFont( font ) {
    assert && assert( font instanceof Font || typeof font === 'string',
      'Fonts provided to setFont should be a Font object or a string in the CSS3 font shortcut format' );

    if ( this._font !== font ) {
      this._font = font;
      this.rebuildRichText();
    }
    return this;
  }

  set font( value ) { this.setFont( value ); }

  /**
   * Returns the current Font
   * @public
   *
   * @returns {Font|string}
   */
  getFont() {
    return this._font;
  }

  get font() { return this.getFont(); }

  /**
   * Sets the fill of our text.
   * @public
   *
   * @param {PaintDef} fill
   * @returns {RichText} - For chaining.
   */
  setFill( fill ) {
    if ( this._fill !== fill ) {
      this._fill = fill;
      this.rebuildRichText();
    }
    return this;
  }

  set fill( value ) { this.setFill( value ); }

  /**
   * Returns the current fill.
   * @public
   *
   * @returns {Font|string}
   */
  getFill() {
    return this._fill;
  }

  get fill() { return this.getFill(); }

  /**
   * Sets the stroke of our text.
   * @public
   *
   * @param {PaintDef} stroke
   * @returns {RichText} - For chaining.
   */
  setStroke( stroke ) {
    if ( this._stroke !== stroke ) {
      this._stroke = stroke;
      this.rebuildRichText();
    }
    return this;
  }

  set stroke( value ) { this.setStroke( value ); }

  /**
   * Returns the current stroke.
   * @public
   *
   * @returns {Font|string}
   */
  getStroke() {
    return this._stroke;
  }

  get stroke() { return this.getStroke(); }

  /**
   * Sets the lineWidth of our text.
   * @public
   *
   * @param {number} lineWidth
   * @returns {RichText} - For chaining.
   */
  setLineWidth( lineWidth ) {
    if ( this._lineWidth !== lineWidth ) {
      this._lineWidth = lineWidth;
      this.rebuildRichText();
    }
    return this;
  }

  set lineWidth( value ) { this.setLineWidth( value ); }

  /**
   * Returns the current lineWidth.
   * @public
   *
   * @returns {number}
   */
  getLineWidth() {
    return this._lineWidth;
  }

  get lineWidth() { return this.getLineWidth(); }

  /**
   * Sets the scale (relative to 1) of any text under subscript (<sub>) elements.
   * @public
   *
   * @param {number} subScale
   * @returs {RichText} - For chaining
   */
  setSubScale( subScale ) {
    assert && assert( typeof subScale === 'number' && isFinite( subScale ) && subScale > 0 );

    if ( this._subScale !== subScale ) {
      this._subScale = subScale;
      this.rebuildRichText();
    }
    return this;
  }

  set subScale( value ) { this.setSubScale( value ); }

  /**
   * Returns the scale (relative to 1) of any text under subscript (<sub>) elements.
   * @public
   *
   * @returns {number}
   */
  getSubScale() {
    return this._subScale;
  }

  get subScale() { return this.getSubScale(); }

  /**
   * Sets the horizontal spacing before any subscript (<sub>) elements.
   * @public
   *
   * @param {number} subXSpacing
   * @returs {RichText} - For chaining
   */
  setSubXSpacing( subXSpacing ) {
    assert && assert( typeof subXSpacing === 'number' && isFinite( subXSpacing ) );

    if ( this._subXSpacing !== subXSpacing ) {
      this._subXSpacing = subXSpacing;
      this.rebuildRichText();
    }
    return this;
  }

  set subXSpacing( value ) { this.setSubXSpacing( value ); }

  /**
   * Returns the horizontal spacing before any subscript (<sub>) elements.
   * @public
   *
   * @returns {number}
   */
  getSubXSpacing() {
    return this._subXSpacing;
  }

  get subXSpacing() { return this.getSubXSpacing(); }

  /**
   * Sets the adjustment offset to the vertical placement of any subscript (<sub>) elements.
   * @public
   *
   * @param {number} subYOffset
   * @returs {RichText} - For chaining
   */
  setSubYOffset( subYOffset ) {
    assert && assert( typeof subYOffset === 'number' && isFinite( subYOffset ) );

    if ( this._subYOffset !== subYOffset ) {
      this._subYOffset = subYOffset;
      this.rebuildRichText();
    }
    return this;
  }

  set subYOffset( value ) { this.setSubYOffset( value ); }

  /**
   * Returns the adjustment offset to the vertical placement of any subscript (<sub>) elements.
   * @public
   *
   * @returns {number}
   */
  getSubYOffset() {
    return this._subYOffset;
  }

  get subYOffset() { return this.getSubYOffset(); }

  /**
   * Sets the scale (relative to 1) of any text under superscript (<sup>) elements.
   * @public
   *
   * @param {number} supScale
   * @returs {RichText} - For chaining
   */
  setSupScale( supScale ) {
    assert && assert( typeof supScale === 'number' && isFinite( supScale ) && supScale > 0 );

    if ( this._supScale !== supScale ) {
      this._supScale = supScale;
      this.rebuildRichText();
    }
    return this;
  }

  set supScale( value ) { this.setSupScale( value ); }

  /**
   * Returns the scale (relative to 1) of any text under superscript (<sup>) elements.
   * @public
   *
   * @returns {number}
   */
  getSupScale() {
    return this._supScale;
  }

  get supScale() { return this.getSupScale(); }

  /**
   * Sets the horizontal spacing before any superscript (<sup>) elements.
   * @public
   *
   * @param {number} supXSpacing
   * @returs {RichText} - For chaining
   */
  setSupXSpacing( supXSpacing ) {
    assert && assert( typeof supXSpacing === 'number' && isFinite( supXSpacing ) );

    if ( this._supXSpacing !== supXSpacing ) {
      this._supXSpacing = supXSpacing;
      this.rebuildRichText();
    }
    return this;
  }

  set supXSpacing( value ) { this.setSupXSpacing( value ); }

  /**
   * Returns the horizontal spacing before any superscript (<sup>) elements.
   * @public
   *
   * @returns {number}
   */
  getSupXSpacing() {
    return this._supXSpacing;
  }

  get supXSpacing() { return this.getSupXSpacing(); }

  /**
   * Sets the adjustment offset to the vertical placement of any superscript (<sup>) elements.
   * @public
   *
   * @param {number} supYOffset
   * @returs {RichText} - For chaining
   */
  setSupYOffset( supYOffset ) {
    assert && assert( typeof supYOffset === 'number' && isFinite( supYOffset ) );

    if ( this._supYOffset !== supYOffset ) {
      this._supYOffset = supYOffset;
      this.rebuildRichText();
    }
    return this;
  }

  set supYOffset( value ) { this.setSupYOffset( value ); }

  /**
   * Returns the adjustment offset to the vertical placement of any superscript (<sup>) elements.
   * @public
   *
   * @returns {number}
   */
  getSupYOffset() {
    return this._supYOffset;
  }

  get supYOffset() { return this.getSupYOffset(); }

  /**
   * Sets the expected cap height (baseline to top of capital letters) as a scale of the detected distance from the
   * baseline to the top of the text bounds.
   * @public
   *
   * @param {number} capHeightScale
   * @returs {RichText} - For chaining
   */
  setCapHeightScale( capHeightScale ) {
    assert && assert( typeof capHeightScale === 'number' && isFinite( capHeightScale ) && capHeightScale > 0 );

    if ( this._capHeightScale !== capHeightScale ) {
      this._capHeightScale = capHeightScale;
      this.rebuildRichText();
    }
    return this;
  }

  set capHeightScale( value ) { this.setCapHeightScale( value ); }

  /**
   * Returns the expected cap height (baseline to top of capital letters) as a scale of the detected distance from the
   * baseline to the top of the text bounds.
   * @public
   *
   * @returns {number}
   */
  getCapHeightScale() {
    return this._capHeightScale;
  }

  get capHeightScale() { return this.getCapHeightScale(); }

  /**
   * Sets the lineWidth of underline lines.
   * @public
   *
   * @param {number} underlineLineWidth
   * @returs {RichText} - For chaining
   */
  setUnderlineLineWidth( underlineLineWidth ) {
    assert && assert( typeof underlineLineWidth === 'number' && isFinite( underlineLineWidth ) && underlineLineWidth > 0 );

    if ( this._underlineLineWidth !== underlineLineWidth ) {
      this._underlineLineWidth = underlineLineWidth;
      this.rebuildRichText();
    }
    return this;
  }

  set underlineLineWidth( value ) { this.setUnderlineLineWidth( value ); }

  /**
   * Returns the lineWidth of underline lines.
   * @public
   *
   * @returns {number}
   */
  getUnderlineLineWidth() {
    return this._underlineLineWidth;
  }

  get underlineLineWidth() { return this.getUnderlineLineWidth(); }

  /**
   * Sets the underline height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   * @public
   *
   * @param {number} underlineHeightScale
   * @returs {RichText} - For chaining
   */
  setUnderlineHeightScale( underlineHeightScale ) {
    assert && assert( typeof underlineHeightScale === 'number' && isFinite( underlineHeightScale ) && underlineHeightScale > 0 );

    if ( this._underlineHeightScale !== underlineHeightScale ) {
      this._underlineHeightScale = underlineHeightScale;
      this.rebuildRichText();
    }
    return this;
  }

  set underlineHeightScale( value ) { this.setUnderlineHeightScale( value ); }

  /**
   * Returns the underline height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   * @public
   *
   * @returns {number}
   */
  getUnderlineHeightScale() {
    return this._underlineHeightScale;
  }

  get underlineHeightScale() { return this.getUnderlineHeightScale(); }

  /**
   * Sets the lineWidth of strikethrough lines.
   * @public
   *
   * @param {number} strikethroughLineWidth
   * @returs {RichText} - For chaining
   */
  setStrikethroughLineWidth( strikethroughLineWidth ) {
    assert && assert( typeof strikethroughLineWidth === 'number' && isFinite( strikethroughLineWidth ) && strikethroughLineWidth > 0 );

    if ( this._strikethroughLineWidth !== strikethroughLineWidth ) {
      this._strikethroughLineWidth = strikethroughLineWidth;
      this.rebuildRichText();
    }
    return this;
  }

  set strikethroughLineWidth( value ) { this.setStrikethroughLineWidth( value ); }

  /**
   * Returns the lineWidth of strikethrough lines.
   * @public
   *
   * @returns {number}
   */
  getStrikethroughLineWidth() {
    return this._strikethroughLineWidth;
  }

  get strikethroughLineWidth() { return this.getStrikethroughLineWidth(); }

  /**
   * Sets the strikethrough height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   * @public
   *
   * @param {number} strikethroughHeightScale
   * @returs {RichText} - For chaining
   */
  setStrikethroughHeightScale( strikethroughHeightScale ) {
    assert && assert( typeof strikethroughHeightScale === 'number' && isFinite( strikethroughHeightScale ) && strikethroughHeightScale > 0 );

    if ( this._strikethroughHeightScale !== strikethroughHeightScale ) {
      this._strikethroughHeightScale = strikethroughHeightScale;
      this.rebuildRichText();
    }
    return this;
  }

  set strikethroughHeightScale( value ) { this.setStrikethroughHeightScale( value ); }

  /**
   * Returns the strikethrough height adjustment as a proportion of the detected distance from the baseline to the top of the
   * text bounds.
   * @public
   *
   * @returns {number}
   */
  getStrikethroughHeightScale() {
    return this._strikethroughHeightScale;
  }

  get strikethroughHeightScale() { return this.getStrikethroughHeightScale(); }

  /**
   * Sets the color of links. If null, no fill will be overridden.
   * @public
   *
   * @param {paint} linkFill
   * @returs {RichText} - For chaining
   */
  setLinkFill( linkFill ) {
    if ( this._linkFill !== linkFill ) {
      this._linkFill = linkFill;
      this.rebuildRichText();
    }
    return this;
  }

  set linkFill( value ) { this.setLinkFill( value ); }

  /**
   * Returns the color of links.
   * @public
   *
   * @returns {paint}
   */
  getLinkFill() {
    return this._linkFill;
  }

  get linkFill() { return this.getLinkFill(); }

  /**
   * Sets whether link clicks will call event.handle().
   * @public
   *
   * @param {boolean} linkEventsHandled
   * @returs {RichText} - For chaining
   */
  setLinkEventsHandled( linkEventsHandled ) {
    assert && assert( typeof linkEventsHandled === 'boolean' );

    if ( this._linkEventsHandled !== linkEventsHandled ) {
      this._linkEventsHandled = linkEventsHandled;
      this.rebuildRichText();
    }
    return this;
  }

  set linkEventsHandled( value ) { this.setLinkEventsHandled( value ); }

  /**
   * Returns whether link events will be handled.
   * @public
   *
   * @returns {boolean}
   */
  getLinkEventsHandled() {
    return this._linkEventsHandled;
  }

  get linkEventsHandled() { return this.getLinkEventsHandled(); }

  /**
   * Sets the map of href placeholder => actual href/callback used for links. However if set to true ({boolean}) as a
   * full object, links in the string will not be mapped, but will be directly added.
   * @public
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
   *
   * @param {Object|boolean} links
   * @returs {RichText} - For chaining
   */
  setLinks( links ) {
    assert && assert( links !== false || Object.getPrototypeOf( links ) === Object.prototype );

    if ( this._links !== links ) {
      this._links = links;
      this.rebuildRichText();
    }
    return this;
  }

  set links( value ) { this.setLinks( value ); }

  /**
   * Returns whether link events will be handled.
   * @public
   *
   * @returns {Object}
   */
  getLinks() {
    return this._links;
  }

  get links() { return this.getLinks(); }

  /**
   * Sets the alignment of text (only relevant if there are multiple lines).
   * @public
   *
   * @param {string} align
   * @returns {RichText} - For chaining
   */
  setAlign( align ) {
    assert && assert( align === 'left' || align === 'center' || align === 'right' );

    if ( this._align !== align ) {
      this._align = align;
      this.rebuildRichText();
    }
    return this;
  }

  set align( value ) { this.setAlign( value ); }

  /**
   * Returns the current alignment of the text (only relevant if there are multiple lines).
   * @public
   *
   * @returns {string}
   */
  getAlign() {
    return this._align;
  }

  get align() { return this.getAlign(); }

  /**
   * Sets the leading (spacing between lines)
   * @public
   *
   * @param {number} leading
   * @returns {RichText} - For chaining
   */
  setLeading( leading ) {
    assert && assert( typeof leading === 'number' && isFinite( leading ) );

    if ( this._leading !== leading ) {
      this._leading = leading;
      this.rebuildRichText();
    }
    return this;
  }

  set leading( value ) { this.setLeading( value ); }

  /**
   * Returns the leading (spacing between lines)
   * @public
   *
   * @returns {number}
   */
  getLeading() {
    return this._leading;
  }

  get leading() { return this.getLeading(); }

  /**
   * Sets the line wrap width for the text (or null if none is desired). Lines longer than this length will wrap
   * automatically to the next line.
   * @public
   *
   * @param {number|null} lineWrap - If it's a number, it should be greater than 0.
   * @returns {RichText} - For chaining
   */
  setLineWrap( lineWrap ) {
    assert && assert( lineWrap === null || ( typeof lineWrap === 'number' && isFinite( lineWrap ) && lineWrap > 0 ) );

    if ( this._lineWrap !== lineWrap ) {
      this._lineWrap = lineWrap;
      this.rebuildRichText();
    }
    return this;
  }

  set lineWrap( value ) { this.setLineWrap( value ); }

  /**
   * Returns the line wrap width.
   * @public
   *
   * @returns {number|null}
   */
  getLineWrap() {
    return this._lineWrap;
  }

  get lineWrap() { return this.getLineWrap(); }

  /**
   * @public
   * @override
   * @param {Object} [options]
   */
  mutate( options ) {
    if ( assert && options.hasOwnProperty( 'text' ) && options.hasOwnProperty( 'textProperty' ) ) {
      assert && assert( options.textProperty.value === options.text, 'If both text and textProperty are provided, then values should match' );
    }
    return super.mutate( options );
  }

  /**
   * Returns a wrapped version of the string with a font specifier that uses the given font object.
   * @public
   *
   * NOTE: Does an approximation of some font values (using <b> or <i>), and cannot force the lack of those if it is
   * included in bold/italic exterior tags.
   *
   * @param {string} str
   * @param {Font} font
   * @returns {string}
   */
  static stringWithFont( str, font ) {
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
   * @public
   *
   * @param {*} element - See himalaya
   * @param {boolean} isLTR
   * @returns {string}
   */
  static himalayaElementToString( element, isLTR ) {
    if ( element.type === 'Text' ) {
      return RichText.contentToString( element.content, isLTR );
    }
    else if ( element.type === 'Element' ) {
      if ( element.tagName === 'span' && element.attributes.dir ) {
        isLTR = element.attributes.dir === 'ltr';
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
   * @public
   *
   * @param {*} element - See himalaya
   * @param {boolean} isLTR
   * @returns {string}
   */
  static himalayaElementToAccessibleString( element, isLTR ) {
    if ( element.type === 'Text' ) {
      return RichText.contentToString( element.content, isLTR );
    }
    else if ( element.type === 'Element' ) {
      if ( element.tagName === 'span' && element.attributes.dir ) {
        isLTR = element.attributes.dir === 'ltr';
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
   * @private
   *
   * See https://github.com/phetsims/scenery-phet/issues/315
   *
   * @param {string} content
   * @param {boolean} isLTR
   * @returns {string}
   */
  static contentToString( content, isLTR ) {
    const unescapedContent = he.decode( content );
    return isLTR ? ( `\u202a${unescapedContent}\u202c` ) : ( `\u202b${unescapedContent}\u202c` );
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @protected
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
RichText.prototype._mutatorKeys = RICH_TEXT_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'RichText', RichText );

const RichTextCleanable = memoize( type => {
  return class extends type {
    /**
     * @public
     *
     * @returns {boolean}
     */
    get isCleanable() {
      return true;
    }

    /**
     * Releases references
     * @public
     */
    clean() {
      // Remove all children (and recursively clean)
      for ( let i = this._children.length - 1; i >= 0; i-- ) {
        const child = this._children[ i ];

        if ( child.isCleanable ) {
          this.removeChild( child );
          child.clean();
        }
      }

      this.matrix = Matrix3.IDENTITY;

      this.freeToPool();
    }
  };
} );

class RichTextElement extends RichTextCleanable( Node ) {
  /**
   * A container of other RichText elements and leaves.
   * @private
   *
   * @param {boolean} isLTR - Whether this container will lay out elements in the left-to-right order (if false, will be
   *                          right-to-left).
   */
  constructor( isLTR ) {
    super();

    this.initialize( isLTR );
  }

  /**
   * Sets up state
   * @public
   *
   * @param {boolean} isLTR
   * @returns {RichTextElement} - Self reference
   */
  initialize( isLTR ) {
    // @private {boolean}
    this.isLTR = isLTR;

    // @protected {number} - The amount of local-coordinate spacing to apply on each side
    this.leftSpacing = 0;
    this.rightSpacing = 0;

    return this;
  }

  /**
   * Adds a child element.
   * @private
   *
   * @param {RichTextElement|RichTextLeaf} element
   * @returns {boolean} - Whether the item was actually added.
   */
  addElement( element ) {

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
        element.left = this.localBounds.right + this.rightSpacing + leftElementSpacing;
        this.rightSpacing = rightElementSpacing;
      }
      else {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `RTL add ${this.leftSpacing} + ${rightElementSpacing}` );
        element.right = this.localBounds.left - this.leftSpacing - rightElementSpacing;
        this.leftSpacing = leftElementSpacing;
      }
      this.addChild( element );
      return true;
    }
    return false;
  }

  /**
   * Adds an amount of spacing to the "before" side.
   * @private
   *
   * @param {number} amount
   */
  addExtraBeforeSpacing( amount ) {
    if ( this.isLTR ) {
      this.leftSpacing += amount;
    }
    else {
      this.rightSpacing += amount;
    }
  }
}

Poolable.mixInto( RichTextElement );

class RichTextLeaf extends RichTextCleanable( Text ) {
  /**
   * A leaf (text) node.
   *
   * @param {string} content
   * @param {boolean} isLTR
   * @param {Font|string} font
   * @param {string} boundsMethod
   * @param {PaintDef} fill
   * @param {PaintDef} stroke
   * @param {number} lineWidth
   */
  constructor( content, isLTR, font, boundsMethod, fill, stroke, lineWidth ) {
    super( '' );

    this.initialize( content, isLTR, font, boundsMethod, fill, stroke, lineWidth );
  }

  /**
   * Set up this text's state
   * @public
   *
   * @param {string} content
   * @param {boolean} isLTR
   * @param {Font|string} font
   * @param {string} boundsMethod
   * @param {PaintDef} fill
   * @param {PaintDef} stroke
   * @param {number} lineWidth
   * @returns {RichTextLeaf} - Self reference
   */
  initialize( content, isLTR, font, boundsMethod, fill, stroke, lineWidth ) {

    // Grab all spaces at the (logical) start
    let whitespaceBefore = '';
    while ( content[ 0 ] === ' ' ) {
      whitespaceBefore += ' ';
      content = content.slice( 1 );
    }

    // Grab all spaces at the (logical) end
    let whitespaceAfter = '';
    while ( content[ content.length - 1 ] === ' ' ) {
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
    // @protected {number}
    this.leftSpacing = isLTR ? spacingBefore : spacingAfter;
    this.rightSpacing = isLTR ? spacingAfter : spacingBefore;

    return this;
  }

  /**
   * Cleans references that could cause memory leaks (as those things may contain other references).
   * @public
   */
  clean() {
    super.clean();

    this.fill = null;
    this.stroke = null;
  }

  /**
   * Whether this leaf will fit in the specified amount of space (including, if required, the amount of spacing on
   * the side).
   * @private
   *
   * @param {number} widthAvailable
   * @param {boolean} hasAddedLeafToLine
   * @param {boolean} isContainerLTR
   */
  fitsIn( widthAvailable, hasAddedLeafToLine, isContainerLTR ) {
    return this.width + ( hasAddedLeafToLine ? ( isContainerLTR ? this.leftSpacing : this.rightSpacing ) : 0 ) <= widthAvailable;
  }
}

Poolable.mixInto( RichTextLeaf );

class RichTextLink extends RichTextCleanable( Node ) {
  /**
   * A link node
   *
   * @param {string} innerContent
   * @param {function|string} href
   */
  constructor( innerContent, href ) {
    super( {
      cursor: 'pointer',
      tagName: 'a'
    } );

    // @private {FireListener|null}
    this.fireListener = null;

    // @private {function|null}
    this.accessibleInputListener = null;

    this.initialize( innerContent, href );
  }

  /**
   * Set up this state
   * @public
   *
   * @param {string} innerContent
   * @param {function|string} href
   * @returns {RichTextLink} - Self reference
   */
  initialize( innerContent, href ) {
    // pdom - open the link in the new tab when activated with a keyboard.
    // also see https://github.com/phetsims/joist/issues/430
    this.innerContent = innerContent;

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
          event.domEvent.preventDefault();

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
            event.domEvent.preventDefault();
          }
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
   * @public
   */
  clean() {
    super.clean();

    this.removeInputListener( this.fireListener );
    this.fireListener = null;
    if ( this.accessibleInputListener ) {
      this.removeInputListener( this.accessibleInputListener );
      this.accessibleInputListener = null;
    }
  }
}

Poolable.mixInto( RichTextLink );

RichText.RichTextIO = new IOType( 'RichTextIO', {
  valueType: RichText,
  supertype: Node.NodeIO,
  documentation: 'The tandem IO Type for the scenery RichText node'
} );

export default RichText;