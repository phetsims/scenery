// Copyright 2016, University of Colorado Boulder

/**
 * Prototype for a cursor that implements the typical navigation strategies of a screen reader.  The output
 * text is meant to be read to a user by the Web Speech API synthesizer.
 *
 * NOTE: This is a prototype for screen reader behavior, and is an initial implementation for
 * a cursor that is to be used together with the web speech API, see
 * https://github.com/phetsims/scenery/issues/538
 *
 * @author Jesse Greenberg
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  // constants
  var SPACE = ' '; // space to insert between words of text content
  var END_OF_DOCUMENT = 'End of Document'; // flag thrown when there is no more content
  var COMMA = ','; // some bits of text content should be separated with a comma for clear synth output
  var LINE_WORD_LENGTH = 15; // number of words read in a single line
  var NEXT = 'NEXT'; // constant that marks the direction of traversal
  var PREVIOUS = 'PREVIOUS'; // constant that marks the direction of tragersal through the DOM

  /**
   * Constructor.
   */
  function Cursor( domElement ) {

    var self = this;

    // the output utterance for the cursor, to be read by the synth and handled in various ways
    // initial output is the document title
    // @public (read-only)
    this.outputUtteranceProperty = new Property( new Utterance( document.title, 'off' ) );

    // @private - a linear representation of the DOM which is navigated by the user
    this.linearDOM = this.getLinearDOMElements( domElement );

    // @private - the active element is element that is under navigation in the parallel DOM
    this.activeElement = null;

    // @private - the active line is the current line being read and navigated with the cursor
    this.activeLine = null;

    // the letter position is the position of the cursor in the active line to support reading on a
    // letter by letter basis.  This is relative to the length of the active line.
    // @private
    this.letterPosition = 0;

    // the positionInLine is the position in words marking the end location of the active line
    // this must be tracked to support content and descriptions longer than 15 words
    // @private
    this.positionInLine = 0;

    // the position of the word in the active line to support navigation on a word by word basis
    // @private
    this.wordPosition = 0;

    // we need to track the mutation observers so that they can be discconnected
    // @private
    this.observers = [];

    // track a keystate in order to handle when multiple key presses happen at once
    // @private
    this.keyState = {};

    // the document will listen for keyboard interactions
    // this listener implements common navigation strategies for a typical screen reader
    //
    // see https://dequeuniversity.com/screenreaders/nvda-keyboard-shortcuts
    // for a list of common navigation strategies
    //
    // TODO: Use this.keyState object instead of referencing the event directly
    document.addEventListener( 'keydown', function( event ) {

      // update the keystate object
      self.keyState[ event.keyCode ] = true;

      // store the output text here
      var outputText;

      // check to see if shift key pressed
      // TODO: we can optionally use the keyState object for this
      var shiftKeyDown = event.shiftKey;

      // direction to navigate through the DOM - usually, holding shift indicates the user wants to travers
      // backwards through the DOM
      var direction = shiftKeyDown ? PREVIOUS : NEXT;

      // the dom can change at any time, make sure that we are reading a copy that is up to date
      self.linearDOM = self.getLinearDOMElements( domElement );

      // update the list of live elements
      self.updateLiveElementList();

      // if the element has an 'application' like behavior, keyboard should be free for the application
      // TODO: This may be insufficient if we need the 'arrow' keys to continue to work for an application role
      if ( self.activeElement && self.activeElement.getAttribute( 'role' ) === 'application' ) {
        return;
      }

      // otherwise, handle all key events here
      if ( self.keyState[ 40 ] && !self.keyState[ 45 ] ) {
        // read the next line on 'down arrow'
        outputText = self.readNextPreviousLine( NEXT );
      }
      else if ( self.keyState[ 38 ] && !self.keyState[ 45 ] ) {
        // read the previous line on 'up arrow'
        outputText = self.readNextPreviousLine( PREVIOUS );
      }
      else if ( self.keyState[ 72 ] ) {
        // read the previous or next headings depending on whether the shift key is pressed
        var headingLevels = [ 'H1', 'H2', 'H3', 'H4', 'H5', 'H6' ];
        outputText = self.readNextPreviousHeading( headingLevels, direction );
      }
      else if ( self.keyState[ 9 ] ) {
        // let the browser naturally handle 'tab' for forms elements and elements with a tabIndex
      }
      else if ( self.keyState[ 39 ] && !self.keyState[ 17 ] ) {
        // read the next character of the active line on 'right arrow'
        outputText = self.readNextPreviousCharacter( NEXT );
      }
      else if ( self.keyState[ 37 ] && !self.keyState[ 17 ] ) {
        // read the previous character on 'left arrow'
        outputText = self.readNextPreviousCharacter( PREVIOUS );
      }
      else if ( self.keyState[ 37 ] && self.keyState[ 17 ] ) {
        // read the previous word on 'control + left arrow'
        outputText = self.readNextPreviousWord( PREVIOUS );
      }
      else if ( self.keyState[ 39 ] && self.keyState[ 17 ] ) {
        // read the next word on 'control + right arrow'
        outputText = self.readNextPreviousWord( NEXT );
      }
      else if ( self.keyState[ 45 ] && self.keyState[ 38 ] ) {
        // repeat the active line on 'insert + up arrow'
        outputText = self.readActiveLine();
      }
      else if ( self.keyState[ 49 ] ) {
        // find the previous/next heading level 1 on '1'
        outputText = self.readNextPreviousHeading( [ 'H1' ], direction );
      }
      else if ( self.keyState[ 50 ] ) {
        // find the previous/next heading level 2 on '2'
        outputText = self.readNextPreviousHeading( [ 'H2' ], direction );
      }
      else if ( self.keyState[ 51 ] ) {
        // find the previous/next heading level 3 on '3'
        outputText = self.readNextPreviousHeading( [ 'H3' ], direction );
      }
      else if ( self.keyState[ 52 ] ) {
        // find the previous/next heading level 4 on '4'
        outputText = self.readNextPreviousHeading( [ 'H4' ], direction );
      }
      else if ( self.keyState[ 53 ] ) {
        // find the previous/next heading level 5 on '5'
        outputText = self.readNextPreviousHeading( [ 'H5' ], direction );
      }
      else if ( self.keyState[ 54 ] ) {
        // find the previous/next heading level 6 on '6'
        outputText = self.readNextPreviousHeading( [ 'H6' ], direction );
      }
      else if ( self.keyState[ 70 ] ) {
        // find the previous/next form element on 'f'
        outputText = self.readNextPreviousFormElement( direction );
      }
      else if ( self.keyState[ 66 ] ) {
        // find the previous/next button element on 'b'
        outputText = self.readNextPreviousButton( direction );
      }
      else if ( self.keyState[ 76 ] ) {
        // find the previous/next list on 'L'
        outputText = self.readNextPreviousList( direction );
      }
      else if ( self.keyState[ 73 ] ) {
        // find the previous/next list item on 'I'
        outputText = self.readNextPreviousListItem( direction );
      }
      else if ( self.keyState[ 45 ] && self.keyState[ 40 ] ) {
        // read entire document on 'insert + down arrow'
        self.readEntireDocument();
      }

      // if the active element is focusable, set the focus to it so that the virtual cursor can
      // directly interact with elements
      if ( self.activeElement && self.isFocusable( self.activeElement ) ) {
        self.activeElement.focus();
      }

      // if the output text is a space, we want it to be read as 'blank' or 'space'
      if ( outputText === SPACE ) {
        outputText = 'space';
      }

      if ( outputText ) {
        // for now, all utterances are off for aria-live
        self.outputUtteranceProperty.set( new Utterance( outputText, 'off' ) );
      }

      // TODO: everything else in https://dequeuniversity.com/screenreaders/nvda-keyboard-shortcuts

    } );

    // update the keystate object on keyup to handle multiple key presses at once
    document.addEventListener( 'keyup', function( event ) {
      self.keyState[ event.keyCode ] = false;
    } );

    // listen for when an element is about to receive focus
    // we are using focusin (and not focus) because we want the event to bubble up the document
    // this will handle both tab navigation AND programatic focus by the simulation
    document.addEventListener( 'focusin', function( event ) {

      // anounce the new focus if it is different from the active element
      if ( event.target !== self.activeElement ) {
        self.activeElement = event.target;

        // so read out all content from aria markup since focus moved via application behavior
        var withApplicationContent = true;
        var outputText = self.getAccessibleText( this.activeElement, withApplicationContent );

        if( outputText ) {
          var liveRole = self.activeElement.getAttribute( 'aria-live' );
          self.outputUtteranceProperty.set( new Utterance( outputText, liveRole ) );
        }
      }
    } );

  }

  scenery.register( 'Cursor', Cursor );

  inherit( Object, Cursor, {

    /**
     * Get all 'element' nodes off the parent element, placing them in an array
     * for easy traversal.  Note that this includes all elements, even those
     * that are 'hidden' or purely for structure.
     *
     * @param  {HTMLElement} domElement - the parent element to linearize
     * @returns {Array.<HTMLElement>}
     * @private
     */
    getLinearDOMElements: function( domElement ) {
      // gets ALL descendent children for the element
      var children = domElement.getElementsByTagName( '*' );

      var linearDOM = [];
      for( var i = 0; i < children.length; i++ ) {
        if( children[i].nodeType === Node.ELEMENT_NODE ) {
          linearDOM[i] = ( children[ i ] );
        }
      }
      return linearDOM;
    },

    /**
     * Get the live role from the DOM element.  If the element is not live, return null.
     *
     * @param  {HTMLElement} domElement
     * @returns {string}
     */
    getLiveRole: function( domElement ) {
      var liveRole = null;

      // collection of all roles that can produce 'live region' behavior
      // see https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
      var roles = [ 'log', 'status', 'alert', 'progressbar', 'marquee', 'timer', 'assertive', 'polite' ];

      roles.forEach( function( role ) {
        if ( domElement.getAttribute( 'aria-live' ) === role || domElement.getAttribute( 'role' ) === role ) {
          liveRole = role;
          return;
        }
      } );

      return liveRole;
    },

    /**
     * Get the next or previous element in the DOM, depending on the desired direction.
     *
     * @param  {[type]} direction - NEXT || PREVIOUS
     * @returns {HTMLElement}
     */
    getNextPreviousElement: function( direction ) {
      if ( !this.activeElement ) {
        this.activeElement = this.linearDOM[ 0 ];
      }

      var searchDelta = direction === 'NEXT' ? 1 : -1;
      var activeIndex = this.linearDOM.indexOf( this.activeElement );

      var nextIndex = activeIndex + searchDelta;
      return this.linearDOM[ nextIndex ];
    },

    /**
     * Get the label for a particular id
     * @param  {string} id
     * @returns {HTMLElement}
     */
    getLabel: function( id ) {
      var labels = document.getElementsByTagName( 'label' );

      // loop through NodeList
      var labelWithId;
      Array.prototype.forEach.call( labels, function( label ) {
        if ( label.getAttribute( 'for' ) ) {
          labelWithId = label;
        }
      } );
      assert && assert( labelWithId, 'No label found for id' );

      return labelWithId;
    },

    /**
     * Get the accessible text from the element.  Depending on the navigation strategy,
     * we may or may not want to include all application content text from the markup.
     *
     * @param  {HTMLElement} element
     * @param  {boolean} withApplicationContent - do you want to include all aria text content?
     * @returns {string}
     */
    getAccessibleText: function( element, withApplicationContent ) {

      // placeholder for the text content that we will build up from the markup
      var textContent = '';

      // if the element is undefined, we have reached the end of the document
      if ( !element ) {
        return END_OF_DOCUMENT;
      }

      // filter out structural elements that do not have accessible text
      if ( element.getAttribute( 'class' ) === 'ScreenView' ) {
        return null;
      }
      if ( element.tagName === 'HEADER' ) {
        // TODO: Headers should have some behavior
        return null;
      }
      if ( element.tagName === 'SECTION' ) {
        // TODO: What do you we do for sections? Read section + aria-labelledby?
        return null;
      }
      if ( element.tagName === 'LABEL' ) {
        // label content is added like 'aria-describedby', do not read this yet
        return null;
      }

      // search up through the ancestors to see if this element should be hidden
      var childElement = element;
      while ( childElement.parentElement ) {
        if ( childElement.getAttribute( 'aria-hidden' ) || childElement.hidden ) {
          return null;
        }
        else { childElement = childElement.parentElement; }
      }

      // search for elements that will have content and should be read
      if ( element.tagName === 'P' ) {
        textContent += element.textContent;
      }
      if( element.tagName === 'H1' ) {
        textContent += 'Heading Level 1, ' + element.textContent;
      }
      if ( element.tagName === 'H2' ) {
        textContent += 'Heading Level 2, ' + element.textContent;
      }
      if ( element.tagName === 'H3' ) {
        textContent += 'Heading Level 3, ' + element.textContent;
      }
      if ( element.tagName === 'UL' ) {
        var listLength = element.children.length;
        textContent += 'List with ' + listLength + ' items';
      }
      if ( element.tagName === 'LI' ) {
        textContent += 'List Item: ' + element.textContent;
      }
      if ( element.tagName === 'BUTTON' ) {
        var buttonLabel = ' Button';
        // check to see if this is a 'toggle' button with the 'aria-pressed' attribute
        if ( element.getAttribute( 'aria-pressed' ) ) {
          var toggleLabel = ' toggle';
          var pressedLabel = ' pressed';
          var notLabel = ' not';

          // insert a comma for readibility of the synth
          toggleLabel += buttonLabel + COMMA;
          if ( element.getAttribute( 'aria-pressed' ) === 'true' ) {
            toggleLabel += pressedLabel;
          }
          else {
            toggleLabel += notLabel + pressedLabel;
          }
          textContent += element.textContent + COMMA + toggleLabel;
        }
        else {
          textContent += element.textContent + buttonLabel;
        }
      }
      if ( element.tagName === 'INPUT' ) {
        if ( element.type === 'reset' ) {
          textContent += element.getAttribute( 'value' ) + ' Button';
        }
        if ( element.type === 'checkbox' ) {
          // the checkbox should have a label - find the correct one
          var checkboxLabel = this.getLabel( element.id );
          var labelContent = checkboxLabel.textContent;

          // describe as a switch if it has the role
          if ( element.getAttribute( 'role' ) === 'switch' ) {
            // required for a checkbox
            var ariaChecked = element.getAttribute( 'aria-checked' );
            if ( ariaChecked ) {
              var switchedString = ( ariaChecked === 'true' ) ? 'On' : 'Off';
              textContent += labelContent + COMMA + SPACE + 'switch' + COMMA + SPACE + switchedString;
            }
            else {
              assert && assert ( false, 'checkbox switch must have aria-checked attribute' );
            }
          }
          else {
            var checkedString = element.checked ? ' Checked' : ' Not Checked';
            textContent += element.textContent + ' Checkbox' + checkedString;
          }
        }
      }

      // if we are in an 'application' style of navigation, we want to add additional information
      // from the markup
      // Order of additions to textContent is important, and is designed to make sense
      // when textContent is read continuously
      // TODO: support more markup!
      if ( withApplicationContent ) {

        // insert a comma at the end of the content to enhance the output of the synth
        if ( textContent.length > 0 ) {
          textContent += COMMA;
        }

        // look for an aria-label
        var ariaLabel = element.getAttribute( 'aria-label' );
        if ( ariaLabel ) {
          textContent += SPACE + ariaLabel + COMMA;
        }

        // look for an aria-labelledBy attribute to see if there is another element in the DOM that
        // describes this one
        var ariaLabelledById = element.getAttribute( 'aria-labelledBy' );
        if ( ariaLabelledById ) {

          var ariaLabelledBy = document.getElementById( ariaLabelledById );
          var ariaLabelledByText = ariaLabelledBy.textContent;

          textContent += SPACE + ariaLabelledByText + COMMA;
        }

        // search up through the ancestors to find if the element has 'application' or 'document' content
        // TODO: Factor out into a searchUp type of function.
        childElement = element;
        var role;
        while ( childElement.parentElement ) {
          role = childElement.getAttribute( 'role' );
          if ( role === 'document' || role === 'application' ) {
            textContent += SPACE + role + COMMA;
            break;
          }
          else { childElement = childElement.parentElement; }
        }

        // check to see if this element has an aria-role
        if ( element.getAttribute( 'role' ) ) {
          role = element.getAttribute( 'role' );
          // TODO handle all the different roles!

          // label if the role is a button
          if ( role === 'button' ) {
            textContent += SPACE + 'Button';
          }
        }

        // check to see if this element is draggable
        if ( element.draggable ) {
          textContent += SPACE + 'draggable' + COMMA;
        }

        // look for aria-grabbed markup to let the user know if the element is grabbed
        if ( element.getAttribute( 'aria-grabbed' ) === 'true' ) {
          textContent += SPACE + 'grabbed' + COMMA;
        }

        // look for an element in the DOM that describes this one
        var ariaDescribedBy = element.getAttribute( 'aria-describedby' );
        if ( ariaDescribedBy ) {
          // the aria spec supports multiple description ID's for a single element
          var descriptionIDs = ariaDescribedBy.split( SPACE );

          var descriptionElement;
          var descriptionText;
          descriptionIDs.forEach( function( descriptionID ) {
            descriptionElement = document.getElementById( descriptionID );
            descriptionText = descriptionElement.textContent;

            textContent += SPACE + descriptionText;
          } );

        }
      }

      // delete the trailing comma if it exists at the end of the textContent
      if( textContent[ textContent.length - 1 ] === ',' ) {
        textContent = textContent.slice( 0, -1 );
      }

      return textContent;
    },

    /**
     * Get the next or previous element in the DOM that has accessible text content, relative to the current
     * active element.
     *
     * @param  {string} direction - NEXT || PREVIOUS
     * @returns {HTMLElement}
     */
    getNextPreviousElementWithAccessibleContent: function( direction ) {
      var accessibleContent;
      while ( !accessibleContent ) {
        // set the selected element to the next element in the DOM
        this.activeElement = this.getNextPreviousElement( direction );
        accessibleContent = this.getAccessibleText( this.activeElement, false );
      }

      return this.activeElement;
    },

    /**
     * Get the next element in the DOM with on of the desired tagNames, types, or roles.  This does not set the active element, it
     * only traverses the document looking for elements.
     *
     * @param  {Array.<string>} roles - list of desired DOM tag names, types, or aria roles
     * @param  {[type]} direction - direction flag for to search through the DOM - NEXT || PREVIOUS
     * @returns {HTMLElement}
     */
    getNextPreviousElementWithRole: function( roles, direction ) {

      var element = null;
      var searchDelta = ( direction === NEXT ) ? 1 : -1;

      // if there is not an active element, use the first element in the DOM.
      if ( !this.activeElement ) {
        this.activeElement = this.linearDOM[ 0 ];
      }

      // start search from the next or previous element and set up the traversal conditions
      var searchIndex = this.linearDOM.indexOf( this.activeElement ) + searchDelta;
      while ( this.linearDOM[ searchIndex ] ) {
        for ( var j = 0; j < roles.length; j++ ) {
          var elementTag = this.linearDOM[ searchIndex ].tagName;
          var elementType = this.linearDOM[ searchIndex ].type;
          var elementRole = this.linearDOM[ searchIndex ].getAttribute( 'role' );
          var searchRole = roles[ j ];
          if ( elementTag === searchRole || elementRole === searchRole || elementType === searchRole ) {
            element = this.linearDOM[ searchIndex ];
            break;
          }
        }
        if ( element ) {
          // we have alread found an element, break out
          break;
        }
        searchIndex += searchDelta;
      }

      return element;
    },

    readNextPreviousLine: function( direction ) {
      var line = '';

      // reset the content letter and word positions because we are reading a new line
      this.letterPosition = 0;
      this.wordPosition = 0;

      // if there is no active element, set to the next element with accessible content
      if ( !this.activeElement ) {
        this.activeElement = this.getNextPreviousElementWithAccessibleContent( direction );
      }

      // get the accessible content for the active element, without any 'application' content, and split into words
      var accessibleText = this.getAccessibleText( this.activeElement, false ).split( SPACE );

      // if traversing backwards, position in line needs be at the start of previous line
      if ( direction === PREVIOUS ) {
        this.positionInLine = this.positionInLine - 2 * LINE_WORD_LENGTH;
      }

      // if there is no content at the line position, it is time to find the next element
      if ( !accessibleText[ this.positionInLine ] ) {
        // reset the position in the line
        this.positionInLine = 0;

        // save the active element in case it needs to be restored
        var previousElement = this.activeElement;

        // update the active element and set the accessible content from this element
        this.activeElement = this.getNextPreviousElementWithAccessibleContent( direction );

        accessibleText = this.getAccessibleText( this.activeElement, false ).split( ' ' );

        // restore the previous active element if we are at the end of the document
        if ( !this.activeElement ) {
          this.activeElement = previousElement;
        }
      }

      // read the next line of the accessible content
      var lineLimit = this.positionInLine + LINE_WORD_LENGTH;
      for( var i = this.positionInLine; i < lineLimit; i++ ) {
        if ( accessibleText[ i ] ) {
          line += accessibleText[ i ];
          this.positionInLine += 1;

          if ( accessibleText[ i + 1 ] ) {
            line += SPACE;
          }
          else {
            // we have reached the end of this content, there are no more words
            // wrap the line position to the end so we can easily read back the previous line
            this.positionInLine += LINE_WORD_LENGTH - this.positionInLine % LINE_WORD_LENGTH;
            break;
          }
        }
      }

      this.activeLine = line;
      return line;
    },

    /**
     * Read the active line without incrementing the word count.
     *
     * @returns {[type]} [description]
     */
    readActiveLine: function() {

      var line = '';

      // if there is no active line, find the next one
      if ( !this.activeLine ) {
        this.activeLine = this.readNextPreviousLine( NEXT );
      }

      // split up the active line into an array of words
      var activeWords = this.activeLine.split( SPACE );

      // read this line of content
      for( var i = 0; i < LINE_WORD_LENGTH; i++ ) {
        if ( activeWords[ i ] ) {
          line += activeWords[ i ];

          if ( activeWords[ i + 1 ] ) {
            // add space if there are more words
            line += SPACE;
          }
          else {
            // we have reached the end of the line, there are no more words
            break;
          }
        }
      }

      return line;
    },

    readNextPreviousWord: function( direction ) {
      // if there is no active line, find the next one
      if ( !this.activeLine ) {
        this.activeLine = this.readNextPreviousLine( direction );
      }

      // split the active line into an array of words
      var activeWords = this.activeLine.split( SPACE );

      // direction dependent variables
      var searchDelta;
      var contentEnd;
      if ( direction === NEXT ) {
        contentEnd = activeWords.length;
        searchDelta = 1;
      }
      else if ( direction === PREVIOUS ) {
        contentEnd = 0;
        searchDelta = -2;
      }

      // if there is no more content, read the next/previous line
      if ( this.wordPosition === contentEnd ) {
        this.activeLine = this.readNextPreviousLine( direction );
      }

      // get the word to read update word position
      var outputText = activeWords[ this.wordPosition ];
      this.wordPosition += searchDelta;

      return outputText;
    },

    /**
     * Read the next or previous heading with one of the levels specified in headingLevels and in the direction
     * specified by the direction flag.
     *
     * @param  {Array.<string>} headingLevels
     * @param  {[type]} direction - direction of traversal through the DOM - NEXT || PREVIOUS
     * @returns {string}
     */
    readNextPreviousHeading: function( headingLevels, direction ) {

      // get the next element in the DOM with one of the above heading levels which has accessible content
      // to read
      var accessibleText;
      var nextElement;

      // track the previous element - if there are no more headings, store it here
      var previousElement;

      while ( !accessibleText ) {
        previousElement = this.activeElement;
        nextElement = this.getNextPreviousElementWithRole( headingLevels, direction );
        this.activeElement = nextElement;
        accessibleText = this.getAccessibleText( nextElement );
      }

      if ( !nextElement ) {
        // restore the active element
        this.activeElement = previousElement;
        // let the user know that there are no more headings at the desired level
        var directionDescriptionString = ( direction === NEXT ) ? 'more' : 'previous';
        if ( headingLevels.length === 1 ) {
          var noNextHeadingString = 'No ' + directionDescriptionString + ' headings at ';

          var headingLevel = headingLevels[ 0 ];
          var levelString = headingLevel === 'H1' ? 'Level 1' :
                            headingLevel === 'H2' ? 'Level 2' :
                            headingLevel === 'H3' ? 'Level 3' :
                            headingLevel === 'H4' ? 'Level 4' :
                            headingLevel === 'H5' ? 'Level 5' :
                            'Level 6';
          return noNextHeadingString + levelString;
        }
        return 'No ' + directionDescriptionString + ' headings';
      }

      // set element as the next active element and return the text
      this.activeElement = nextElement;
      return accessibleText;
    },

    /**
     * Read the next/previous button element.  A button can have the tagname button, have the aria button role, or
     * or have one of the following types: submit, button, reset
     * @param  {string}} direction
     * @returns {HTMLElement}
     */
    readNextPreviousButton: function( direction ) {
      // the following roles should handle 'role=button', 'type=button', 'tagName=BUTTON'
      var roles = [ 'button', 'BUTTON', 'submit', 'reset' ];

      var nextElement;
      var accessibleText;
      var previousElement;

      while( !accessibleText ) {
        previousElement = this.activeElement;
        nextElement = this.getNextPreviousElementWithRole( roles, direction );
        this.activeElement = nextElement;

        // get the accessible text with application descriptions
        accessibleText = this.getAccessibleText( nextElement, true );
      }

      if ( !nextElement ) {
        this.activeElement = previousElement;
        var directionDescriptionString = direction === NEXT ? 'more' : 'previous';
        return 'No ' + directionDescriptionString + ' buttons';
      }

      this.activeElement = nextElement;
      return accessibleText;
    },

    readNextPreviousFormElement: function( direction ) {
      // TODO: support more form elements!
      var tagNames = [ 'INPUT', 'BUTTON' ];
      var ariaRoles = [ 'button' ];
      var roles = tagNames.concat( ariaRoles );

      var nextElement;
      var accessibleText;

      // track the previous element - if there are no more form elements it will need to be restored
      var previousElement;

      while ( !accessibleText ) {
        previousElement = this.activeElement;
        nextElement = this.getNextPreviousElementWithRole( roles, direction );
        this.activeElement = nextElement;

        // get the accessible text with aria descriptions
        accessibleText = this.getAccessibleText( nextElement, true );
      }

      if ( accessibleText === END_OF_DOCUMENT ) {
        this.activeElement = previousElement;
        var directionDescriptionString = direction === NEXT ? 'next' : 'previous';
        return 'No ' + directionDescriptionString + ' form field';
      }

      this.activeElement = nextElement;
      return accessibleText;
    },

    readNextPreviousListItem: function( direction ) {
      if ( !this.activeElement ) {
        this.activeElement = this.getNextPreviousElementWithAccessibleContent( direction );
      }

      var accessibleText;

      // if we are inside of a list, get the next peer, or find the next list
      var parentElement = this.activeElement.parentElement;
      if ( parentElement.tagName === 'UL' || parentElement.tagName === 'OL' ) {

        var searchDelta = direction === NEXT ? 1 : -1;

        // Array.prototype must be used on the NodeList
        var searchIndex = Array.prototype.indexOf.call( parentElement.children, this.activeElement ) + searchDelta;

        while ( parentElement.children[ searchIndex ] ) {
          accessibleText = this.getAccessibleText( parentElement.children[ searchIndex ] );
          if ( accessibleText ) {
            this.activeElement = parentElement.children[ searchIndex ];
            break;
          }
          searchIndex += searchDelta;
        }

        if ( !accessibleText ) {
          // there was no accessible text in the list items, so read the next / previous list
          accessibleText = this.readNextPreviousList( direction );
        }
      }
      else {
        // not inside of a list, so read the next/previous one and its first item
        accessibleText = this.readNextPreviousList( direction );
      }

      if ( !accessibleText ) {
        var directionDescriptionString = ( direction === NEXT ) ? 'more' : 'previous';
        return 'No ' + directionDescriptionString + ' list items';
      }

      return accessibleText;
    },

    readNextPreviousList: function( direction ) {
      if ( !this.activeElement ) {
        this.activeElement = this.getNextPreviousElementWithAccessibleContent( direction );
      }

      // if we are inside of a list already, step out of it to begin searching there
      var parentElement = this.activeElement.parentElement;
      var activeElement;
      if ( parentElement.tagName === 'UL' || parentElement.tagName === 'OL' ) {
        // save the previous active element - if there are no more lists, this should not change
        activeElement = this.activeElement;

        this.activeElement = parentElement;
      }

      var listElement = this.getNextPreviousElementWithRole( [ 'UL', 'OL' ], direction );

      if ( !listElement ) {

        // restore the previous active element
        if ( activeElement ) {
          this.activeElement = activeElement;
        }

        // let the user know that there are no more lists and move to the next element
        var directionDescriptionString = direction === NEXT ? 'more' : 'previous';
        return 'No ' + directionDescriptionString + ' lists';
      }

      // get the content from the list element
      var listText = this.getAccessibleText( listElement );

      // include the content from the first item in the list
      var itemText = '';
      var firstItem = listElement.children[ 0 ];
      if ( firstItem ) {
        itemText = this.getAccessibleText( firstItem );
        this.activeElement = firstItem;
      }

      return listText + ', ' + itemText;
    },

    readNextPreviousCharacter: function( direction ) {
      // if there is no active line, find the next one
      if ( !this.activeLine ) {
        this.activeLine = this.readNextPreviousLine( NEXT );
      }

      // directional dependent variables
      var contentEnd;
      var searchDelta;
      var normalizeDirection;
      if ( direction === NEXT ) {
        contentEnd = this.activeLine.length;
        searchDelta = 1;
        normalizeDirection = 0;
      }
      else if (direction === PREVIOUS ) {
        // for backwards traversal, read from two characters behind
        contentEnd = 2;
        searchDelta = -1;
        normalizeDirection = -2;
      }

      // if we are at the end of the content, read the next/previous line
      if ( this.letterPosition === contentEnd ) {
        this.activeLine = this.readNextPreviousLine( direction );

        // if reading backwards, letter position should be at the end of the active line
        this.letterPosition = this.activeLine.length;
      }

      // get the letter to read and increment the letter position
      var outputText = this.activeLine[ this.letterPosition + normalizeDirection ];
      this.letterPosition += searchDelta;

      return outputText;
    },

    /**
     * Update the list of elements, and add Mutation Observers to each one.  MutationObservers
     * provide a way to listen to changes in the DOM,
     * see https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver
     */
    updateLiveElementList: function() {

      var self = this;

      // remove all previous observers
      // TODO: only update the observer list if necessary
      for( var i = 0 ; i < this.observers.length; i++ ) {
        if (this.observers[ i ] ) {
          this.observers[ i ].disconnect();
        }
      }

      // clear the list of observers
      this.observers = [];

      // search through the DOM, looking for elements with a 'live region' attribute
      for ( i = 0; i < this.linearDOM.length; i++ ) {
        var domElement = this.linearDOM[ i ];
        var liveRole = self.getLiveRole( domElement );

        if( liveRole ) {
          var mutationObserverCallback = function( mutations ) {
            mutations.forEach( function( mutation ) {
              var liveRole;
              var mutatedElement = mutation.target;

              // look for the type of live role that is associated with this mutation
              // if the target has no live attribute, search through the element's ancestors to find the attribute
              while( !liveRole ) {
                liveRole = self.getLiveRole( mutatedElement );
                mutatedElement = mutatedElement.parentElement;
              }

              // we only care about nodes added
              if ( mutation.addedNodes[ 0 ] ) {
                var updatedText = mutation.addedNodes[ 0 ].data;
                self.outputUtteranceProperty.set( new Utterance( updatedText, liveRole ) );
              }
            } );
          };

          // create a mutation observer for this live element
          var observer = new MutationObserver( function( mutations ) {
            mutationObserverCallback( mutations );
          } );

          // listen for changes to the subtree in case children of the aria-live parent change their textContent
          var observerConfig = { childList: true, subtree: true };

          observer.observe( domElement, observerConfig );
          self.observers.push( observer );
        }
      }
    },

    /**
     * Read continuously from the current active element.  Accessible content is read by reader with a 'polite'
     * utterance so that new text is added to the queue line by line.
     *
     * TODO: If the read is cancelled, the active element should be set appropriately.
     *
     * @returns {string}
     */
    readEntireDocument: function() {

      var liveRole = 'polite';
      var outputText = this.getAccessibleText( this.activeElement );
      var activeElement = this.activeElement;

      while ( outputText !== END_OF_DOCUMENT ) {
        activeElement = this.activeElement;
        outputText = this.readNextPreviousLine( NEXT );

        if ( outputText === END_OF_DOCUMENT ) {
          this.activeElement = activeElement;
        }
        this.outputUtteranceProperty.set( new Utterance( outputText, liveRole ) );
      }
    },

    /**
     * Return true if the element is focusable.  A focusable element has a tab index, is a
     * form element, or has a role which adds it to the navigation order.
     *
     * TODO: Populate with the rest of the focusable elements.
     * @param  {HTMLElement} domElement
     * @returns {boolean}
     */
    isFocusable: function( domElement ) {
      // list of attributes and tag names which should be in the navigation order
      // TODO: more roles!
      var focusableRoles = [ 'tabindex', 'BUTTON', 'INPUT' ];

      var focusable = false;
      focusableRoles.forEach( function( role ) {

        if ( domElement.getAttribute( role ) ) {
          focusable = true;
          return;
        }
        else if ( domElement.tagName === role ) {
          focusable = true;
          return;
        }
      } );
      return focusable;
    }
  } );

  /**
   * Create an experimental type to create unique utterances for the reader.
   * Type is simply a collection of text and a priority for aria-live that
   * lets the reader know whether to queue the next utterance or cancel it in the order.
   *
   * TODO: This is where we could deviate from traditional screen reader behavior. For instance, instead of
   * just liveRole, perhaps we should have a liveIndex that specifies order of the live update? We may also
   * need additional flags here for the reader.
   *
   * @param {string} text - the text to be read as the utterance for the synth
   * @param {string} liveRole - see https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
   */
  function Utterance( text, liveRole ) {

    this.text = text;
    this.liveRole = liveRole;

  }

  inherit( Object, Utterance );

  return Cursor;
} );