// Copyright 2017, University of Colorado Boulder

/**
 * Displays rich text by interpreting the input text as HTML, supporting a limited set of tags that prevent any
 * security vulnerabilities. It does this by parsing the input HTML and splitting it into multiple Text children
 * recursively.
 *
 * NOTE: Encoding HTML entities is required, and malformed HTML is not accepted.
 *
 * It supports the following markup and features:
 * - <a href="{{placeholder}}"> for links (pass in { links: { placeholder: ACTUAL_HREF } })
 * - <b> and <strong> for bold text
 * - <i> and <em> for italic text
 * - <sub> and <sup> for subscripts / superscripts
 * - <u> for underlined text
 * - <s> for strikethrough text
 * - <font> tags with attributes color="cssString", face="familyString", size="cssSize"
 * - <span> tags with a dir="ltr" / dir="rtl" attribute
 * - Unicode bidirectional marks (present in PhET strings) for full RTL support
 *
 * Examples from the scenery-phet demo:
 *
 * new RichText( 'RichText can have <b>bold</b> and <i>italic</i> text.' ),
 * new RichText( 'Can do H<sub>2</sub>O (A<sub>sub</sub> and A<sup>sup</sup>), or nesting: x<sup>2<sup>2</sup></sup>' ),
 * new RichText( 'Additionally: <font color="blue">color</font>, <font size="30px">sizes</font>, <font face="serif">faces</font>, <s>strikethrough</s>, and <u>underline</u>' ),
 * new RichText( 'These <b><em>can</em> <u><font color="red">be</font> mixed<sup>1</sup></u></b>.' ),
 * new RichText( '\u202aHandles bidirectional text: \u202b<font color="#0a0">مقابض</font> النص ثنائي <b>الاتجاه</b><sub>2</sub>\u202c\u202c' )
 * new RichText( '\u202b\u062a\u0633\u062a (\u0632\u0628\u0627\u0646)\u202c' ),
 * new RichText( 'HTML entities need to be escaped, like &amp; and &lt;.' ),
 * new RichText( 'Supports <a href="{{phetWebsite}}"><em>links</em> with <b>markup</b></a>.', {
 *   links: {
 *     phetWebsite: 'https://phet.colorado.edu'
 *   }
 * } ),
 * new RichText( 'Or also <a href="https://phet.colorado.edu">links directly in the string</a>.', {
 *   links: true
 * } ),
 * new RichText( 'Links not found <a href="{{bogus}}">are ignored</a> for security.' )
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var ButtonListener = require( 'SCENERY/input/ButtonListener' );
  var Color = require( 'SCENERY/util/Color' );
  var extendDefined = require( 'PHET_CORE/extendDefined' );
  var Font = require( 'SCENERY/util/Font' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Line = require( 'SCENERY/nodes/Line' );
  var Node = require( 'SCENERY/nodes/Node' );
  var scenery = require( 'SCENERY/scenery' );
  var Text = require( 'SCENERY/nodes/Text' );
  var Tandem = require( 'TANDEM/Tandem' );
  var TRichText = require( 'SCENERY/nodes/TRichText' );

  // constants
  var RICH_TEXT_OPTION_KEYS = [
    'font',
    'fill',
    'stroke',
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
    'text'
  ];

  var DEFAULT_FONT = new Font( {
    size: 20
  } );

  // Tags that should be included in accessibleLabel, see https://github.com/phetsims/joist/issues/430
  var ACCESSIBLE_TAGS = [
    'b', 'strong', 'i', 'em', 'sub', 'sup', 'u', 's'
  ];

  /**
   * @public
   * @constructor
   * @extends Node
   * @mixes Events
   *
   * @param {string|number} text
   * @param {Object} [options] - RichText-specific options are documented in RICH_TEXT_OPTION_KEYS above, and can be
   *                             provided along-side options for Node.
   */
  function RichText( text, options ) {

    // @private {string} - Set by mutator
    this._text = '';

    // @private {Font}
    this._font = DEFAULT_FONT;

    // @private {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
    this._fill = '#000000';
    this._stroke = null;

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

    // @private {Object|boolean}
    this._links = {};

    Node.call( this );

    // @private {Node} - So we don't mess with child nodes that may get added to us.
    this.richTextContainer = new Node();
    this.addChild( this.richTextContainer );

    options = extendDefined( {
      fill: '#000000',
      text: text,
      tandem: Tandem.tandemOptional(),
      phetioType: TRichText
    }, options );

    this.mutate( options );
  }

  scenery.register( 'RichText', RichText );

  return inherit( Node, RichText, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: RICH_TEXT_OPTION_KEYS.concat( Node.prototype._mutatorKeys ),

    /**
     * When called, will rebuild the node structure for this RichText
     * @private
     */
    rebuildRichText: function() {
      this.richTextContainer.removeAllChildren();

      // Turn bidirectional marks into explicit elements, so that the nesting is applied correctly.
      var mappedText = this._text.replace( /\u202a/g, '<span dir="ltr">' )
        .replace( /\u202b/g, '<span dir="rtl">' )
        .replace( /\u202c/g, '</span>' );

      // Start appending all top-level elements
      var rootElements = himalaya.parse( mappedText );
      for ( var i = 0; i < rootElements.length; i++ ) {
        this.appendElement( this.richTextContainer, rootElements[ i ], this._font, this._fill, true );
      }
    },

    /**
     * @private
     *
     * @param {Node} containerNode - The node where child elements should be placed
     * @param {*} element - See Himalaya's element specification
     *                      (https://github.com/andrejewski/himalaya/blob/master/text/ast-spec-v0.md)
     * @param {Font|string} font - The font to apply at this level
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} fill - Fill to apply
     * @param {boolean} isLTR - True if LTR, false if RTL (handles RTL text properly)
     */
    appendElement: function( containerNode, element, font, fill, isLTR ) {
      var self = this;

      var nextSideName = isLTR ? 'left' : 'right';
      var previousSideName = isLTR ? 'right' : 'left';

      // If our container has content, we want to know where to add to that on the right side.
      var x = isFinite( containerNode.localBounds[ previousSideName ] ) ? containerNode.localBounds[ previousSideName ] : 0;

      // {Node|Text} - The main Node for the element that we are adding
      var node;

      // If we're a leaf
      if ( element.type === 'Text' ) {
        node = new Text( RichText.contentToString( element.content, isLTR ), {
          font: font,
          fill: fill,
          stroke: this._stroke
        } );
      }
      // Otherwise presumably an element with content
      else if ( element.type === 'Element' ) {
        node = new Node();

        if ( element.tagName === 'a' ) {
          var href = element.attributes.href;
          if ( this._links !== true ) {
            if ( href.indexOf( '{{' ) === 0 && href.indexOf( '}}' ) === href.length - 2 ) {
              href = this._links[ href.slice( 2, -2 ) ];
            }
            else {
              href = null;
            }
          }
          if ( href ) {
            if ( this._linkFill !== null ) {
              fill = this._linkFill; // Link color
            }
            node.cursor = 'pointer';
            node.addInputListener( new ButtonListener( {
              fire: function( event ) {
                self._linkEventsHandled && event.handle();
                var newWindow = window.open( href, '_blank' ); // open in a new window/tab
                newWindow.focus();
              }
            } ) );
            // a11y - open the link in the new tab when activated with a keyboard.
            // also see https://github.com/phetsims/joist/issues/430
            node.tagName = 'a';
            node.accessibleLabel = RichText.himalayaElementToAccessibleString( element, isLTR );
            node.setAccessibleAttribute( 'href', href );
            node.setAccessibleAttribute( 'target', '_blank' );
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
          node.x += ( isLTR ? 1 : -1 ) * this._subXSpacing;
          node.y += this._subYOffset;
        }
        // Superscript
        else if ( element.tagName === 'sup' ) {
          node.scale( this._supScale );
          node.x += ( isLTR ? 1 : -1 ) * this._supXSpacing;
          node.y += this._supYOffset;
        }
        // Font (color/face/size attributes)
        else if ( element.tagName === 'font' ) {
          if ( element.attributes.color ) {
            fill = new Color( element.attributes.color );
          }
          if ( element.attributes.face ) {
            font = font.copy( {
              family: element.attributes.face
            } );
          }
          if ( element.attributes.size ) {
            font = font.copy( {
              size: element.attributes.size
            } );
          }
        }
        // Span (dir attribute)
        else if ( element.tagName === 'span' ) {
          if ( element.attributes.dir ) {
            assert && assert( element.attributes.dir === 'ltr' || element.attributes.dir === 'rtl',
              'Span dir attributes should be ltr or rtl.' );
            isLTR = element.attributes.dir === 'ltr';
          }
        }

        // Process children
        for ( var i = 0; i < element.children.length; i++ ) {
          this.appendElement( node, element.children[ i ], font, fill, isLTR );
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
            node.centerY = new Text( 'X', { font: font } ).top * this._capHeightScale;
          }
        }
        // Underline
        else if ( element.tagName === 'u' ) {
          var underlineY = -node.top * this._underlineHeightScale;
          if ( isFinite( node.top ) ) {
            node.addChild( new Line( node.localBounds.left, underlineY, node.localBounds.right, underlineY, {
              stroke: fill,
              lineWidth: this._underlineLineWidth
            } ) );
          }
        }
        // Strikethrough
        else if ( element.tagName === 's' ) {
          var strikethroughY = node.top * this._strikethroughHeightScale;
          if ( isFinite( node.top ) ) {
            node.addChild( new Line( node.localBounds.left, strikethroughY, node.localBounds.right, strikethroughY, {
              stroke: fill,
              lineWidth: this._strikethroughLineWidth
            } ) );
          }
        }
      }

      // Only add/position the content if it has finite bounds (ignoring empty elements)
      if ( isFinite( node.width ) ) {
        node[ nextSideName ] = x;
        containerNode.addChild( node );
      }
    },

    /**
     * Sets the text displayed by our node.
     * @public
     *
     * NOTE: Encoding HTML entities is required, and malformed HTML is not accepted.
     *
     * @param {string|number} text - The text to display. If it's a number, it will be cast to a string
     * @returns {RichText} - For chaining
     */
    setText: function( text ) {
      assert && assert( text !== null && text !== undefined, 'Text should be defined and non-null. Use the empty string if needed.' );
      assert && assert( typeof text === 'number' || typeof text === 'string', 'text should be a string or number' );

      // cast it to a string (for numbers, etc., and do it before the change guard so we don't accidentally trigger on non-changed text)
      text = '' + text;

      if ( text !== this._text ) {
        this._text = text;
        this.rebuildRichText();
      }
      return this;
    },
    set text( value ) { this.setText( value ); },

    /**
     * Returns the text displayed by our node.
     * @public
     *
     * @returns {string}
     */
    getText: function() {
      return this._text;
    },
    get text() { return this.getText(); },

    /**
     * Sets the font of our node.
     * @public
     *
     * @param {Font|string} font
     * @returns {RichText} - For chaining.
     */
    setFont: function( font ) {
      assert && assert( font instanceof Font || typeof font === 'string',
        'Fonts provided to setFont should be a Font object or a string in the CSS3 font shortcut format' );

      if ( this._font !== font ) {
        this._font = font;
        this.rebuildRichText();
      }
      return this;
    },
    set font( value ) { this.setFont( value ); },

    /**
     * Returns the current Font
     * @public
     *
     * @returns {Font|string}
     */
    getFont: function() {
      return this._font;
    },
    get font() { return this.getFont(); },

    /**
     * Sets the fill of our text.
     * @public
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} fill
     * @returns {RichText} - For chaining.
     */
    setFill: function( fill ) {
      if ( this._fill !== fill ) {
        this._fill = fill;
        this.rebuildRichText();
      }
      return this;
    },
    set fill( value ) { this.setFill( value ); },

    /**
     * Returns the current fill.
     * @public
     *
     * @returns {Font|string}
     */
    getFill: function() {
      return this._fill;
    },
    get fill() { return this.getFill(); },

    /**
     * Sets the stroke of our text.
     * @public
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} stroke
     * @returns {RichText} - For chaining.
     */
    setStroke: function( stroke ) {
      if ( this._stroke !== stroke ) {
        this._stroke = stroke;
        this.rebuildRichText();
      }
      return this;
    },
    set stroke( value ) { this.setStroke( value ); },

    /**
     * Returns the current stroke.
     * @public
     *
     * @returns {Font|string}
     */
    getStroke: function() {
      return this._stroke;
    },
    get stroke() { return this.getStroke(); },

    /**
     * Sets the scale (relative to 1) of any text under subscript (<sub>) elements.
     * @public
     *
     * @param {number} subScale
     * @returs {RichText} - For chaining
     */
    setSubScale: function( subScale ) {
      if ( this._subScale !== subScale ) {
        this._subScale = subScale;
        this.rebuildRichText();
      }
      return this;
    },
    set subScale( value ) { this.setSubScale( value ); },

    /**
     * Returns the scale (relative to 1) of any text under subscript (<sub>) elements.
     * @public
     *
     * @returns {number}
     */
    getSubScale: function() {
      return this._subScale;
    },
    get subScale() { return this.getSubScale(); },

    /**
     * Sets the horizontal spacing before any subscript (<sub>) elements.
     * @public
     *
     * @param {number} subXSpacing
     * @returs {RichText} - For chaining
     */
    setSubXSpacing: function( subXSpacing ) {
      if ( this._subXSpacing !== subXSpacing ) {
        this._subXSpacing = subXSpacing;
        this.rebuildRichText();
      }
      return this;
    },
    set subXSpacing( value ) { this.setSubXSpacing( value ); },

    /**
     * Returns the horizontal spacing before any subscript (<sub>) elements.
     * @public
     *
     * @returns {number}
     */
    getSubXSpacing: function() {
      return this._subXSpacing;
    },
    get subXSpacing() { return this.getSubXSpacing(); },

    /**
     * Sets the adjustment offset to the vertical placement of any subscript (<sub>) elements.
     * @public
     *
     * @param {number} subYOffset
     * @returs {RichText} - For chaining
     */
    setSubYOffset: function( subYOffset ) {
      if ( this._subYOffset !== subYOffset ) {
        this._subYOffset = subYOffset;
        this.rebuildRichText();
      }
      return this;
    },
    set subYOffset( value ) { this.setSubYOffset( value ); },

    /**
     * Returns the adjustment offset to the vertical placement of any subscript (<sub>) elements.
     * @public
     *
     * @returns {number}
     */
    getSubYOffset: function() {
      return this._subYOffset;
    },
    get subYOffset() { return this.getSubYOffset(); },

    /**
     * Sets the scale (relative to 1) of any text under superscript (<sup>) elements.
     * @public
     *
     * @param {number} supScale
     * @returs {RichText} - For chaining
     */
    setSupScale: function( supScale ) {
      if ( this._supScale !== supScale ) {
        this._supScale = supScale;
        this.rebuildRichText();
      }
      return this;
    },
    set supScale( value ) { this.setSupScale( value ); },

    /**
     * Returns the scale (relative to 1) of any text under superscript (<sup>) elements.
     * @public
     *
     * @returns {number}
     */
    getSupScale: function() {
      return this._supScale;
    },
    get supScale() { return this.getSupScale(); },

    /**
     * Sets the horizontal spacing before any superscript (<sup>) elements.
     * @public
     *
     * @param {number} supXSpacing
     * @returs {RichText} - For chaining
     */
    setSupXSpacing: function( supXSpacing ) {
      if ( this._supXSpacing !== supXSpacing ) {
        this._supXSpacing = supXSpacing;
        this.rebuildRichText();
      }
      return this;
    },
    set supXSpacing( value ) { this.setSupXSpacing( value ); },

    /**
     * Returns the horizontal spacing before any superscript (<sup>) elements.
     * @public
     *
     * @returns {number}
     */
    getSupXSpacing: function() {
      return this._supXSpacing;
    },
    get supXSpacing() { return this.getSupXSpacing(); },

    /**
     * Sets the adjustment offset to the vertical placement of any superscript (<sup>) elements.
     * @public
     *
     * @param {number} supYOffset
     * @returs {RichText} - For chaining
     */
    setSupYOffset: function( supYOffset ) {
      if ( this._supYOffset !== supYOffset ) {
        this._supYOffset = supYOffset;
        this.rebuildRichText();
      }
      return this;
    },
    set supYOffset( value ) { this.setSupYOffset( value ); },

    /**
     * Returns the adjustment offset to the vertical placement of any superscript (<sup>) elements.
     * @public
     *
     * @returns {number}
     */
    getSupYOffset: function() {
      return this._supYOffset;
    },
    get supYOffset() { return this.getSupYOffset(); },

    /**
     * Sets the expected cap height (baseline to top of capital letters) as a scale of the detected distance from the
     * baseline to the top of the text bounds.
     * @public
     *
     * @param {number} capHeightScale
     * @returs {RichText} - For chaining
     */
    setCapHeightScale: function( capHeightScale ) {
      if ( this._capHeightScale !== capHeightScale ) {
        this._capHeightScale = capHeightScale;
        this.rebuildRichText();
      }
      return this;
    },
    set capHeightScale( value ) { this.setCapHeightScale( value ); },

    /**
     * Returns the expected cap height (baseline to top of capital letters) as a scale of the detected distance from the
     * baseline to the top of the text bounds.
     * @public
     *
     * @returns {number}
     */
    getCapHeightScale: function() {
      return this._capHeightScale;
    },
    get capHeightScale() { return this.getCapHeightScale(); },

    /**
     * Sets the lineWidth of underline lines.
     * @public
     *
     * @param {number} underlineLineWidth
     * @returs {RichText} - For chaining
     */
    setUnderlineLineWidth: function( underlineLineWidth ) {
      if ( this._underlineLineWidth !== underlineLineWidth ) {
        this._underlineLineWidth = underlineLineWidth;
        this.rebuildRichText();
      }
      return this;
    },
    set underlineLineWidth( value ) { this.setUnderlineLineWidth( value ); },

    /**
     * Returns the lineWidth of underline lines.
     * @public
     *
     * @returns {number}
     */
    getUnderlineLineWidth: function() {
      return this._underlineLineWidth;
    },
    get underlineLineWidth() { return this.getUnderlineLineWidth(); },

    /**
     * Sets the underline height adjustment as a proportion of the detected distance from the baseline to the top of the
     * text bounds.
     * @public
     *
     * @param {number} underlineHeightScale
     * @returs {RichText} - For chaining
     */
    setUnderlineHeightScale: function( underlineHeightScale ) {
      if ( this._underlineHeightScale !== underlineHeightScale ) {
        this._underlineHeightScale = underlineHeightScale;
        this.rebuildRichText();
      }
      return this;
    },
    set underlineHeightScale( value ) { this.setUnderlineHeightScale( value ); },

    /**
     * Returns the underline height adjustment as a proportion of the detected distance from the baseline to the top of the
     * text bounds.
     * @public
     *
     * @returns {number}
     */
    getUnderlineHeightScale: function() {
      return this._underlineHeightScale;
    },
    get underlineHeightScale() { return this.getUnderlineHeightScale(); },

    /**
     * Sets the lineWidth of strikethrough lines.
     * @public
     *
     * @param {number} strikethroughLineWidth
     * @returs {RichText} - For chaining
     */
    setStrikethroughLineWidth: function( strikethroughLineWidth ) {
      if ( this._strikethroughLineWidth !== strikethroughLineWidth ) {
        this._strikethroughLineWidth = strikethroughLineWidth;
        this.rebuildRichText();
      }
      return this;
    },
    set strikethroughLineWidth( value ) { this.setStrikethroughLineWidth( value ); },

    /**
     * Returns the lineWidth of strikethrough lines.
     * @public
     *
     * @returns {number}
     */
    getStrikethroughLineWidth: function() {
      return this._strikethroughLineWidth;
    },
    get strikethroughLineWidth() { return this.getStrikethroughLineWidth(); },

    /**
     * Sets the strikethrough height adjustment as a proportion of the detected distance from the baseline to the top of the
     * text bounds.
     * @public
     *
     * @param {number} strikethroughHeightScale
     * @returs {RichText} - For chaining
     */
    setStrikethroughHeightScale: function( strikethroughHeightScale ) {
      if ( this._strikethroughHeightScale !== strikethroughHeightScale ) {
        this._strikethroughHeightScale = strikethroughHeightScale;
        this.rebuildRichText();
      }
      return this;
    },
    set strikethroughHeightScale( value ) { this.setStrikethroughHeightScale( value ); },

    /**
     * Returns the strikethrough height adjustment as a proportion of the detected distance from the baseline to the top of the
     * text bounds.
     * @public
     *
     * @returns {number}
     */
    getStrikethroughHeightScale: function() {
      return this._strikethroughHeightScale;
    },
    get strikethroughHeightScale() { return this.getStrikethroughHeightScale(); },

    /**
     * Sets the color of links. If null, no fill will be overridden.
     * @public
     *
     * @param {paint} linkFill
     * @returs {RichText} - For chaining
     */
    setLinkFill: function( linkFill ) {
      if ( this._linkFill !== linkFill ) {
        this._linkFill = linkFill;
        this.rebuildRichText();
      }
      return this;
    },
    set linkFill( value ) { this.setLinkFill( value ); },

    /**
     * Returns the color of links.
     * @public
     *
     * @returns {paint}
     */
    getLinkFill: function() {
      return this._linkFill;
    },
    get linkFill() { return this.getLinkFill(); },

    /**
     * Sets whether link clicks will call event.handle().
     * @public
     *
     * @param {boolean} linkEventsHandled
     * @returs {RichText} - For chaining
     */
    setLinkEventsHandled: function( linkEventsHandled ) {
      if ( this._linkEventsHandled !== linkEventsHandled ) {
        this._linkEventsHandled = linkEventsHandled;
        this.rebuildRichText();
      }
      return this;
    },
    set linkEventsHandled( value ) { this.setLinkEventsHandled( value ); },

    /**
     * Returns whether link events will be handled.
     * @public
     *
     * @returns {boolean}
     */
    getLinkEventsHandled: function() {
      return this._linkEventsHandled;
    },
    get linkEventsHandled() { return this.getLinkEventsHandled(); },

    /**
     * Sets the map of href placeholder => actual href used for links. However if set to true ({boolean}) as a full
     * object, links in the string will not be mapped, but will be directly added.
     * @public
     *
     * For instance, the default is to map hrefs for security purposes:
     *
     * new RichText( '<a href="alink">content</a>', {
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
     * See https://github.com/phetsims/scenery-phet/issues/316 for more information.
     *
     * @param {Object|boolean} links
     * @returs {RichText} - For chaining
     */
    setLinks: function( links ) {
      if ( this._links !== links ) {
        this._links = links;
        this.rebuildRichText();
      }
      return this;
    },
    set links( value ) { this.setLinks( value ); },

    /**
     * Returns whether link events will be handled.
     * @public
     *
     * @returns {Object}
     */
    getLinks: function() {
      return this._links;
    },
    get links() { return this.getLinks(); }
  }, {
    /**
     * Stringifies an HTML subtree defined by the given element.
     * @public
     *
     * @param {*} element - See himalaya
     * @param {boolean} isLTR
     * @returns {string}
     */
    himalayaElementToString: function( element, isLTR ) {
      if ( element.type === 'Text' ) {
        return RichText.contentToString( element.content, isLTR );
      }
      else if ( element.type === 'Element' ) {
        if ( element.tagName === 'span' && element.attributes.dir ) {
          isLTR = element.attributes.dir === 'ltr';
        }

        // Process children
        return element.children.map( function( child ) {
          return RichText.himalayaElementToString( child, isLTR );
        } ).join( '' );
      }
      else {
        return '';
      }
    },

    /**
     * Stringifies an HTML subtree defined by the given element, but removing certain tags that we don't need for
     * accessibility (like <a>, <font>, <span>, etc.), and adding in tags we do want (see ACCESSIBLE_TAGS).
     * @public
     *
     * @param {*} element - See himalaya
     * @param {boolean} isLTR
     * @returns {string}
     */
    himalayaElementToAccessibleString: function( element, isLTR ) {
      if ( element.type === 'Text' ) {
        return RichText.contentToString( element.content, isLTR );
      }
      else if ( element.type === 'Element' ) {
        if ( element.tagName === 'span' && element.attributes.dir ) {
          isLTR = element.attributes.dir === 'ltr';
        }

        // Process children
        var content = element.children.map( function( child ) {
          return RichText.himalayaElementToAccessibleString( child, isLTR );
        } ).join( '' );

        if ( _.includes( ACCESSIBLE_TAGS, element.tagName ) ) {
          return '<' + element.tagName + '>' + content + '</' + element.tagName + '>';
        }
        else {
          return content;
        }
      }
      else {
        return '';
      }
    },

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
    contentToString: function( content, isLTR ) {
      var unescapedContent = _.unescape( content );
      return isLTR ? ( '\u202a' + unescapedContent + '\u202c' ) : ( '\u202b' + unescapedContent + '\u202c' );
    }
  } );
} );
