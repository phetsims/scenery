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
 * - <br> for explicit line breaks
 * - Unicode bidirectional marks (present in PhET strings) for full RTL support
 *
 * Examples from the scenery-phet demo:
 *
 * new RichText( 'RichText can have <b>bold</b> and <i>italic</i> text.' ),
 * new RichText( 'Can do H<sub>2</sub>O (A<sub>sub</sub> and A<sup>sup</sup>), or nesting: x<sup>2<sup>2</sup></sup>' ),
 * new RichText( 'Additionally: <font color="blue">color</font>, <font size="30px">sizes</font>, <font face="serif">faces</font>, <s>strikethrough</s>, and <u>underline</u>' ),
 * new RichText( 'These <b><em>can</em> <u><font color="red">be</font> mixed<sup>1</sup></u></b>.' ),
 * new RichText( '\u202aHandles bidirectional text: \u202b<font color="#0a0">مقابض</font> النص ثنائي <b>الاتجاه</b><sub>2</sub>\u202c\u202c' ),
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
 *     new RichText( 'Supposedly RichText supports line wrapping. Here is a lineWrap of 300, which should probably wrap multiple times here', { lineWrap: 300 })
 *   ]
 * } )
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
  var VStrut = require( 'SCENERY/nodes/VStrut' );

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
    'align',
    'leading',
    'lineWrap',
    'text'
  ];

  var DEFAULT_FONT = new Font( {
    size: 20
  } );

  // Tags that should be included in accessibleLabel, see https://github.com/phetsims/joist/issues/430
  var ACCESSIBLE_TAGS = [
    'b', 'strong', 'i', 'em', 'sub', 'sup', 'u', 's'
  ];

  var LineBreakState = {
    COMPLETE: 'COMPLETE',
    INCOMPLETE: 'INCOMPLETE',
    NONE: 'NONE'
  };

  var scratchText = new scenery.Text( '' );

  /**
   * @public
   * @constructor
   * @extends Node
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

    // @private {string}
    this._align = 'left'; // 'left', 'center', or 'right'

    // @private {number}
    this._leading = 0;

    // @private {number|null}
    this._lineWrap = null;

    // @private {Array.<{ element: {*}, node: {Node}, href: {string} }>}
    this._linkItems = [];

    // @private {boolean}
    this._hasAddedLeafToLine = false;

    Node.call( this );

    // @private {Node} - Normal layout of lines
    this.lineContainer = new Node( {} );
    this.addChild( this.lineContainer );

    // @private {Node} - Extracted nodes from links go here instead
    this.linkContainer = new Node( {} );
    this.addChild( this.linkContainer );

    options = extendDefined( {
      fill: '#000000',
      text: text,
      tandem: Tandem.tandemOptional(),
      phetioType: TRichText
    }, options );

    this.mutate( options );
  }

  scenery.register( 'RichText', RichText );

  inherit( Node, RichText, {
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
      var self = this;

      this.lineContainer.removeAllChildren();
      this.linkContainer.removeAllChildren();


      // Bail early, particularly if we are being constructed.
      if ( this._text === '' ) {
        return;
      }

      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'RichText#' + this.id + ' rebuild' );
      sceneryLog && sceneryLog.RichText && sceneryLog.push();

      // Turn bidirectional marks into explicit elements, so that the nesting is applied correctly.
      var mappedText = this._text.replace( /\u202a/g, '<span dir="ltr">' )
        .replace( /\u202b/g, '<span dir="rtl">' )
        .replace( /\u202c/g, '</span>' );

      // Start appending all top-level elements
      var rootElements = himalaya.parse( mappedText );

      // Clear out link items, as we'll need to reconstruct them later
      this._linkItems.length = 0;

      var widthAvailable = this._lineWrap === null ? Number.POSITIVE_INFINITY : this._lineWrap;

      var isRootLTR = true;

      var currentLine = new RichTextElement( isRootLTR );
      this._hasAddedLeafToLine = false; // notify that if nothing has been added, the first leaf always gets added.

      while ( rootElements.length ) {
        var element = rootElements[ 0 ];

        var currentLineWidth = currentLine.bounds.isValid() ? currentLine.width : 0;

        var lineBreakState = this.appendElement( currentLine, element, this._font, this._fill, isRootLTR, widthAvailable - currentLineWidth );
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'lineBreakState: ' + lineBreakState );

        if ( lineBreakState !== LineBreakState.NONE ) {
          if ( currentLine.bounds.isValid() ) {
            sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Adding line due to lineBreak' );
            this.appendLine( currentLine );
          }
          else {
            // If there's a blank line, add in a strut
            this.appendLine( new VStrut( new Text( ' ', { font: this._font } ).height ) );
          }
          currentLine = new RichTextElement( isRootLTR );
          this._hasAddedLeafToLine = false;
        }
        if ( lineBreakState !== LineBreakState.INCOMPLETE ) {
          sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Finished root element' );
          rootElements.splice( 0, 1 );
        }
      }
      if ( currentLine.bounds.isValid() ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Adding final line' );
        this.appendLine( currentLine );
      }

      this.alignLines();

      // Handle regrouping of links
      while ( this._linkItems.length ) {
        // Close over the href and other references
        (function(){
          var linkElement = self._linkItems[ 0 ].element;
          var href = self._linkItems[ 0 ].href;
          var i;

          // Find all nodes that are for the same link
          var nodes = [];
          for ( i = self._linkItems.length - 1; i >= 0; i-- ) {
            var item = self._linkItems[ i ];
            if ( item.element === linkElement ) {
              nodes.push( item.node );
              self._linkItems.splice( i, 1 );
            }
          }

          // a11y - open the link in the new tab when activated with a keyboard.
          // also see https://github.com/phetsims/joist/issues/430
          var rootNode = new Node( {
            cursor: 'pointer',
            tagName: 'a',
            accessibleLabel: linkElement.accessibleLabel
          } );

          if ( typeof href === 'function' ) {
            rootNode.addInputListener( new ButtonListener( {
              fire: href
            } ) );
            rootNode.setAccessibleAttribute( 'href', '#' );
            rootNode.addAccessibleInputListener( {
              click: function( event ) {
                event.preventDefault();

                href();
              }
            } );
          }
          else {
            rootNode.addInputListener( new ButtonListener( {
              fire: function( event ) {
                self._linkEventsHandled && event.handle();
                var newWindow = window.open( href, '_blank' ); // open in a new window/tab
                newWindow.focus();
              }
            } ) );
            rootNode.setAccessibleAttribute( 'href', href );
            rootNode.setAccessibleAttribute( 'target', '_blank' );
          }

          self.linkContainer.addChild( rootNode );

          // Detach the node from its location, adjust its transform, and reattach under the link
          for ( i = 0; i < nodes.length; i++ ) {
            var node = nodes[ i ];
            var matrix = node.getUniqueTrailTo( self.lineContainer ).getMatrix();
            node.detach();
            node.matrix = matrix;
            rootNode.addChild( node );
          }
        })();
      }

      // Clear them out afterwards, for memory purposes
      this._linkItems.length = 0;

      sceneryLog && sceneryLog.RichText && sceneryLog.pop();
    },

    /**
     * Appends a finished line, applying any necessary leading.
     * @private
     *
     * @param {Node} lineNode
     */
    appendLine: function( lineNode ) {
      // Apply leading
      if ( this.lineContainer.bounds.isValid() ) {
        lineNode.top = this.lineContainer.bottom + this._leading;

        // This ensures RTL lines will still be laid out properly with the main origin (handled by alignLines later)
        lineNode.left = 0;
      }

      this.lineContainer.addChild( lineNode );
    },

    /**
     * Aligns all lines attached to the lineContainer.
     * @private
     */
    alignLines: function() {
      var coordinateName = this._align === 'center' ? 'centerX' : this._align;

      var ideal = this.lineContainer[ coordinateName ];
      for ( var i = 0; i < this.lineContainer.getChildrenCount(); i++ ) {
        this.lineContainer.getChildAt( i )[ coordinateName ] = ideal;
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
     * @param {number} widthAvailable - How much width we have available before forcing a line break
     * @returns {LineBreakState} - Whether a line break was reached
     */
    appendElement: function( containerNode, element, font, fill, isLTR, widthAvailable ) {
      var lineBreakState = LineBreakState.NONE;

      // {Node|Text} - The main Node for the element that we are adding
      var node;

      var extraSpacing = 0;

      // If we're a leaf
      if ( element.type === 'Text' ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'appending leaf: ' + element.content );
        sceneryLog && sceneryLog.RichText && sceneryLog.push();

        node = new RichTextLeaf( element.content, isLTR, font, fill, this._stroke );

        // Handle wrapping if required
        if ( node.width > widthAvailable ) {
          var words = element.content.split( ' ' );

          sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Overflow leafAdded:' + this._hasAddedLeafToLine + ', words: ' + words.length );

          // If we need to add something (and there is only a single word), then add it
          if ( this._hasAddedLeafToLine || words.length > 1 ) {
            sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Skipping words' );

            var skippedWords = [];
            var success = false;
            skippedWords.unshift( words.pop() ); // We didn't fit with the last one!

            while ( words.length ) {
              node = new RichTextLeaf( words.join( ' ' ), isLTR, font, fill, this._stroke );

              // If we haven't added anything to the line and we are down to the first word, we need to just add it.
              if ( node.width > widthAvailable && ( this._hasAddedLeafToLine || words.length > 1 ) ) {
                sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Skipping word ' + words[ words.length - 1 ] );
                skippedWords.unshift( words.pop() );
              }
              else {
                sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Success with ' + words.join( ' ' ) );
                success = true;
                break;
              }
            }

            // If we haven't added anything yet to this line, we'll permit the overflow
            if ( success ) {
              lineBreakState = LineBreakState.INCOMPLETE;
              element.content = skippedWords.join( ' ' );
              sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'Remaining content: ' + element.content );
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
        if ( element.tagName === 'br' ) {
          sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'manual line break' );
          return LineBreakState.COMPLETE;
        }
        // Span (dir attribute)
        else if ( element.tagName === 'span' ) {
          if ( element.attributes.dir ) {
            assert && assert( element.attributes.dir === 'ltr' || element.attributes.dir === 'rtl',
              'Span dir attributes should be ltr or rtl.' );
            isLTR = element.attributes.dir === 'ltr';
          }
        }

        node = new RichTextElement( isLTR );

        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'appending element' );
        sceneryLog && sceneryLog.RichText && sceneryLog.push();

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
            // Don't overwrite only accessibleLabels once things have been "torn down"
            if ( !element.accessibleLabel ) {
              element.accessibleLabel = RichText.himalayaElementToAccessibleString( element, isLTR );
            }
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
          extraSpacing += this._subXSpacing;
          node.y += this._subYOffset;
        }
        // Superscript
        else if ( element.tagName === 'sup' ) {
          node.scale( this._supScale );
          extraSpacing += this._supXSpacing;
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

        // If we've added extra spacing, we'll need to subtract it off of our available width
        widthAvailable -= extraSpacing;
        var scale = node.getScaleVector().x;

        // Process children
        while ( lineBreakState === LineBreakState.NONE && element.children.length ) {
          var widthBefore = node.bounds.isValid() ? node.width : 0;

          var childElement = element.children[ 0 ];
          lineBreakState = this.appendElement( node, childElement, font, fill, isLTR, widthAvailable / scale );
          if ( lineBreakState !== LineBreakState.INCOMPLETE ) {
            element.children.splice( 0, 1 );
          }

          var widthAfter = node.bounds.isValid() ? node.width : 0;

          // Remove the amount of width taken up by the
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
        sceneryLog && sceneryLog.RichText && sceneryLog.pop();
      }

      containerNode.addElement( node );

      return lineBreakState;
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
    get links() { return this.getLinks(); },

    /**
     * Sets the alignment of text (only relevant if there are multiple lines).
     * @public
     *
     * @param {string} align
     * @returns {RichText} - For chaining
     */
    setAlign: function( align ) {
      assert && assert( align === 'left' || align === 'center' || align === 'right' );

      if ( this._align !== align ) {
        this._align = align;
        this.rebuildRichText();
      }
      return this;
    },
    set align( value ) { this.setAlign( value ); },

    /**
     * Returns the current alignment of the text (only relevant if there are multiple lines).
     * @public
     *
     * @returns {string}
     */
    getAlign: function() {
      return this._align;
    },
    get align() { return this.getAlign(); },

    /**
     * Sets the leading (spacing between lines)
     * @public
     *
     * @param {number} leading
     * @returns {RichText} - For chaining
     */
    setLeading: function( leading ) {
      assert && assert( typeof leading === 'number' && isFinite( leading ) );

      if ( this._leading !== leading ) {
        this._leading = leading;
        this.rebuildRichText();
      }
      return this;
    },
    set leading( value ) { this.setLeading( value ); },

    /**
     * Returns the leading (spacing between lines)
     * @public
     *
     * @returns {number}
     */
    getLeading: function() {
      return this._leading;
    },
    get leading() { return this.getLeading(); },

    /**
     * Sets the line wrap width for the text (or null if none is desired). Lines longer than this length will wrap
     * automatically to the next line.
     * @public
     *
     * @param {number|null} lineWrap - If it's a number, it should be greater than 0.
     * @returns {RichText} - For chaining
     */
    setLineWrap: function( lineWrap ) {
      assert && assert( lineWrap === null || ( typeof lineWrap === 'number' && isFinite( lineWrap ) && lineWrap > 0 ) );

      if ( this._lineWrap !== lineWrap ) {
        this._lineWrap = lineWrap;
        this.rebuildRichText();
      }
      return this;
    },
    set lineWrap( value ) { this.setLineWrap( value ); },

    /**
     * Returns the line wrap width.
     * @public
     *
     * @returns {number|null}
     */
    getLineWrap: function() {
      return this._lineWrap;
    },
    get lineWrap() { return this.getLineWrap(); }
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

  /**
   * A container of other RichText elements and leaves.
   * @constructor
   * @private
   *
   * @param {boolean} isLTR - Whether this container will lay out elements in the left-to-right order (if false, will be
   *                          right-to-left).
   */
  function RichTextElement( isLTR ) {
    Node.call( this );

    // @private {boolean}
    this.isLTR = isLTR;

    // @protected {number} - The amount of local-coordinate spacing to apply on each side
    this.leftSpacing = 0;
    this.rightSpacing = 0;
  }

  inherit( Node, RichTextElement, {
    /**
     * Adds a child element.
     * @private
     *
     * @param {RichTextElement|RichTextLeaf} element
     */
    addElement: function( element ) {

      var hadChild = this.children.length > 0;
      var hasElement = element.width > 0;

      // May be at a different scale, which we need to handle
      var elementScale = element.getScaleVector().x;
      var leftElementSpacing = element.leftSpacing * elementScale;
      var rightElementSpacing = element.rightSpacing * elementScale;

      // If there is nothing, than no spacing should be handled
      if ( !hadChild && !hasElement ) {
        return;
      }
      else if ( !hadChild ) {
        if ( this.isLTR ) {
          element.left = 0;
          this.rightSpacing = rightElementSpacing;
        }
        else {
          element.right = 0;
          this.leftSpacing = leftElementSpacing;
        }
        this.addChild( element );
      }
      else if ( !hasElement ) {
        if ( this.isLTR ) {
          this.rightSpacing += leftElementSpacing + rightElementSpacing;
        }
        else {
          this.leftSpacing += leftElementSpacing + rightElementSpacing;
        }
      }
      else {
        if ( this.isLTR ) {
          element.left = this.localBounds.right + this.rightSpacing + leftElementSpacing;
          this.rightSpacing = rightElementSpacing;
        }
        else {
          element.right = this.localBounds.left + this.leftSpacing + rightElementSpacing;
          this.leftSpacing = leftElementSpacing;
        }
        this.addChild( element );
      }
    }
  } );

  /**
   * A leaf (text) node.
   * @constructor
   *
   * @param {string} content
   * @param {boolean} isLTR
   * @param {Font|string} font
   * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} fill
   * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} stroke
   */
  function RichTextLeaf( content, isLTR, font, fill, stroke ) {

    // Grab all spaces at the (logical) start
    var whitespaceBefore = '';
    while ( content[ 0 ] === ' ' ) {
      whitespaceBefore += ' ';
      content = content.slice( 1 );
    }

    // Grab all spaces at the (logical) end
    var whitespaceAfter = '';
    while ( content[ content.length - 1 ] === ' ' ) {
      whitespaceAfter = ' ';
      content = content.slice( 0, content.length - 1 );
    }

    Text.call( this, RichText.contentToString( content, isLTR ), {
      font: font,
      fill: fill,
      stroke: stroke
    } );

    var spacingBefore = whitespaceBefore.length ? scratchText.setText( whitespaceBefore ).setFont( font ).width : 0;
    var spacingAfter = whitespaceAfter.length ? scratchText.setText( whitespaceAfter ).setFont( font ).width : 0;

    // Turn logical spacing into directional
    // @protected {number}
    this.leftSpacing = isLTR ? spacingBefore : spacingAfter;
    this.rightSpacing = isLTR ? spacingAfter : spacingBefore;
  }

  inherit( Text, RichTextLeaf );

  return RichText;
} );
