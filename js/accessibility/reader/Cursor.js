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

  /**
   * Constructor.
   */
  function Cursor( domElement ) {

    var thisCursor = this;

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
      thisCursor.keyState[ event.keyCode ] = true;

      // store the output text here
      var outputText;

      // check to see if shift key pressed
      // TODO: we can optionally use the keyState object for this
      var shiftKeyDown = event.shiftKey;

      // the dom can change at any time, make sure that we are reading a copy that is up to date
      thisCursor.linearDOM = thisCursor.getLinearDOMElements( domElement );

      // update the list of live elements
      thisCursor.updateLiveElementList();

      // handle all of the various navigation strategies here
      if ( thisCursor.keyState[ 40 ] && !thisCursor.keyState[ 45 ] ) {
        // read the next line on 'down arrow'
        outputText = thisCursor.readNextLine();
      }
      else if ( thisCursor.keyState[ 38 ] && !thisCursor.keyState[ 45 ] ) {
        // read the previous line on 'up arrow'
        outputText = thisCursor.readPreviousLine();
      }
      else if ( thisCursor.keyState[ 72 ] ) {
        // read the previous or next headings depending on whether the shift key is pressed
        var headingLevels = [ 'H1', 'H2', 'H3', 'H4', 'H5', 'H6' ];
        outputText = shiftKeyDown ? thisCursor.readPreviousHeading( headingLevels ) : thisCursor.readNextHeading( headingLevels );
      }
      else if ( thisCursor.keyState[ 9 ] ) {
        // let the browser naturally handle 'tab' for forms elements and elements with a tabIndex
      }
      else if ( thisCursor.keyState[ 39 ] && !thisCursor.keyState[ 17 ] ) {
        // read the next character of the active line on 'right arrow'
        outputText = thisCursor.readNextCharacter();
      }
      else if ( thisCursor.keyState[ 37 ] && !thisCursor.keyState[ 17 ] ) {
        // read the previous character on 'left arrow'
        outputText = thisCursor.readPreviousCharacter();
      }
      else if ( thisCursor.keyState[ 37 ] && thisCursor.keyState[ 17 ] ) {
        // read the previous word on 'control + left arrow'
        outputText = thisCursor.readPreviousWord();
      }
      else if ( thisCursor.keyState[ 39 ] && thisCursor.keyState[ 17 ] ) {
        // read the next word on 'control + right arrow'
        outputText = thisCursor.readNextWord();
      }
      else if ( thisCursor.keyState[ 45 ] && thisCursor.keyState[ 38 ] ) {
        // repeat the active line on 'insert + up arrow'
        outputText = thisCursor.readActiveLine(); 
      }
      else if ( thisCursor.keyState[ 49 ] ) {
        // find the previous/next heading level 1 on '1'
        var level1 = [ 'H1' ];
        outputText = shiftKeyDown ? thisCursor.readPreviousHeading( level1 ) : thisCursor.readNextHeading( level1 );
      }
      else if ( thisCursor.keyState[ 50 ] ) {
        // find the previous/next heading level 2 on '2'
        var level2 = [ 'H2' ];
        outputText = shiftKeyDown ? thisCursor.readPreviousHeading( level2 ) : thisCursor.readNextHeading( level2 );
      }
      else if ( thisCursor.keyState[ 51 ] ) {
        // find the previous/next heading level 3 on '3'
        var level3 = [ 'H3' ];
        outputText = shiftKeyDown ? thisCursor.readPreviousHeading( level3 ) : thisCursor.readNextHeading( level3 );
      }
      else if ( thisCursor.keyState[ 52 ] ) {
        // find the previous/next heading level 4 on '4'
        var level4 = [ 'H4' ];
        outputText = shiftKeyDown ? thisCursor.readPreviousHeading( level4 ) : thisCursor.readNextHeading( level4 );
      }
      else if ( thisCursor.keyState[ 53 ] ) {
        // find the previous/next heading level 5 on '5'
        var level5 = [ 'H5' ];
        outputText = shiftKeyDown ? thisCursor.readPreviousHeading( level5 ) : thisCursor.readNextHeading( level5 );
      }
      else if ( thisCursor.keyState[ 54 ] ) {
        // find the previous/next heading level 6 on '6'
        var level6 = [ 'H6' ];
        outputText = shiftKeyDown ? thisCursor.readPreviousHeading( level6 ) : thisCursor.readNextHeading( level6 );
      }
      else if ( thisCursor.keyState[ 70 ] ) {
        // find the previous/next form element on 'f'
        outputText = shiftKeyDown ? thisCursor.readPreviousFormElement() : thisCursor.readNextFormElement();
      }
      else if ( thisCursor.keyState[ 76 ] ) {
        // find the previous/next list on 'L'
        outputText = shiftKeyDown ? thisCursor.readPreviousList() : thisCursor.readNextList();
      }
      else if ( thisCursor.keyState[ 73 ] ) {
        // find the previous/next list item on 'I'
        outputText = shiftKeyDown ? thisCursor.readPreviousListItem() : thisCursor.readNextListItem();
      }
      else if ( thisCursor.keyState[ 45 ] && thisCursor.keyState[ 40 ] ) {
        // read entire document on 'insert + down arrow'
        thisCursor.readEntireDocument();
      }

      // if the active element is focusable, set the focus to it so that the virtual cursor can
      // directly interact with elements
      if( thisCursor.activeElement && thisCursor.isFocusable( thisCursor.activeElement ) ) {
        thisCursor.activeElement.focus();
      }

      // if the output text is a space, we want it to be read as 'blank' or 'space'
      if ( outputText === SPACE ) {
        outputText = 'space';
      }

      if ( outputText ) {
        // for now, all utterances are off for aria-live
       thisCursor.outputUtteranceProperty.set( new Utterance( outputText, 'off' ) );
      }

      // TODO: everything else in https://dequeuniversity.com/screenreaders/nvda-keyboard-shortcuts
      
    } );

    // update the keystate object on keyup to handle multiple key presses at once
    document.addEventListener( 'keyup', function( event ) {
      thisCursor.keyState[ event.keyCode ] = false;
    } );

    // listen for when an element is about to receive focus
    // we are using focusin and not focus because we want the event to bubble up to the document
    // this will handle both tab navigation and programatic focus by the simulation
    document.addEventListener( 'focusin', function( event ) {

      // anounce the new focus if it is different from the active element
      if ( event.target !== thisCursor.activeElement ) {
        thisCursor.activeElement = event.target;

        // so read out all content from aria markup since focus moved via application behavior
        var withApplicationContent = true;
        var outputText = thisCursor.getAccessibleText( this.activeElement, withApplicationContent );

        if( outputText ) {
          var liveRole = thisCursor.activeElement.getAttribute( 'aria-live' );
          thisCursor.outputUtteranceProperty.set( new Utterance( outputText, liveRole ) );
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
     * @param  {DOMElement} domElement - the parent element to linearize
     * @return {Array.<DOMElement>}
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
     * @param  {DOMElement} domElement
     * @return {string}
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
     * Get the next element in the linearized DOM relative to the active element
     * 
     * @return {DOMElement}
     */
    getNextElement: function() {
      if ( !this.activeElement ) {
        this.activeElement = this.linearDOM[ 0 ];
      }

      var activeIndex = this.linearDOM.indexOf( this.activeElement );
      var nextIndex = activeIndex + 1;

      return this.linearDOM[ nextIndex ];
    },

    /**
     * Get the previous element in the linearized DOM relative to the active element
     * 
     * @return {DOMElement} 
     */
    getPreviousElement: function() {
      if( !this.activeElement ) {
        this.activeElement = this.linearDOM[ 0 ];
      }

      var activeIndex = this.linearDOM.indexOf( this.activeElement );
      var previousIndex = activeIndex - 1;

      return this.linearDOM[ previousIndex ];
    },

    /**
     * Get the accessible text from the element.  Depending on the navigation strategy,
     * we may or may not want to include all application content text from the markup.
     * 
     * @param  {DOMElement} element
     * @param  {boolean} withApplicationContent - do you want to include all aria text content? 
     * @return {string}             
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
      if ( element.getAttribute( 'aria-hidden' ) || element.hidden ) {
        return null;
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
        textContent += element.textContent + ' Button';
      }
      if ( element.tagName === 'INPUT' ) {
        if ( element.type === 'reset' ) {
          textContent += element.getAttribute( 'value' ) + ' Button';
        }
        if ( element.type === 'checkbox' ) {
          var checkedString = element.checked ? ' Checked' : ' Not Checked';
          textContent += element.textContent + ' Checkbox' + checkedString;
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

        // check to see if this element is draggable
        if ( element.draggable ) {
          textContent += SPACE + 'Draggable' + COMMA;
        }

        // look for aria-grabbed markup to let the user know if the element is grabbed
        if ( element.getAttribute( 'aria-grabbed' ) ) {
          textContent += SPACE + 'Grabbed' + COMMA;
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
     * Get the next element in the DOM that has accessible text content, relative to the 
     * active element.
     * 
     * @return {DOMElement}
     */
    getNextElementWithAccessibleContent: function() {
      var accessibleContent;
      while( !accessibleContent ) {
        // set the selected element to the next element in the DOM
        this.activeElement = this.getNextElement();
        accessibleContent = this.getAccessibleText( this.activeElement, false );
      }

      // the active element is already set to the next element with accessible content,
      // but return it for completeness
      return this.activeElement;
    },

    /**
     * Get the previous element in the DOM that has accessible text content, relative to the 
     * active element.
     *
     * @return {DOMElement}
     */
    getPreviousElementWithAccessibleContent: function() {
      var accessibleContent;
      while ( !accessibleContent ) {
        // set the selected element to the previous element in the DOM
        this.activeElement = this.getPreviousElement();
        accessibleContent = this.getAccessibleText( this.activeElement, false );
      }

      // the active element is already set to the next element with accessible
      // content, but return it for completeness.
      return this.activeElement;
    },

    /**
     * Get the next element in the DOM with one of the specified tag names, relative to the 
     * currently active element.
     * 
     * @param  {Array.<string>} tagNames - HTML tag name
     * @return {DOMElement}
     */
    getNextElementWithTagName: function( tagNames ) {

      var element = null;

      // if there is not an active element, set to the first element in the DOM
      if ( !this.activeElement ) {
        this.activeElement = this.linearDOM[ 0 ];
      }

      // start search from the next element in the DOM
      var searchIndex = this.linearDOM.indexOf( this.activeElement ) + 1;

      // search through the remaining DOM for an element with a tagName specified
      // in tagNames
      for ( var i = searchIndex; i < this.linearDOM.length; i++ ) {
        for ( var j = 0; j < tagNames.length; j++ ) {
          if ( this.linearDOM[ i ].tagName === tagNames[ j ] ) {
            element = this.linearDOM[ i ];
            this.activeElement = element;
            break;
          }
        }
        if ( element ) {
          // go ahead and break out if we found something
          break;
        }
      }

      // we have alread set the active element to the element with the tag 
      // name but return for completeness
      return element;
    },

    /**
     * Get the previous element in the DOM
     * 
     * @param  {Array.<string>} tagNames - array of possible tag names
     * @return {DOMElement}
     */
    getPreviousElementWithTagName: function( tagNames ) {
      var element = null;

      // if there is no active element, start at the beginning of the DOM
      if ( !this.activeElement ) {
        this.activeElement = this.linearDOM[ 0 ];
      }

      // start the search at the previous element in the DOM
      var searchIndex = this.linearDOM.indexOf( this.activeElement ) - 1;

      // search backwards through the DOM for an element with a tagname
      for ( var i = searchIndex; i >= 0; i-- ) {
        for( var j = 0; j < tagNames.length; j++ ) {
          if ( this.linearDOM[ i ].tagName === tagNames[ j ] ) {
            element = this.linearDOM[ i ];
            this.activeElement = element;
            break;
          }
        }
        if ( element ) {
          // break if we have found something already
          break;
        }
      }

      return element;
    },

    /**
     * Read the next line of content from the DOM.  A line is a string of words with length
     * limitted by LINE_WORD_LENGTH.
     * 
     * @return {string}
     */
    readNextLine: function() {

      var line = '';

      // reset the content letter position because we have a new line
      this.letterPosition = 0;
      this.wordPosition = 0;

      // if there is no active element, set to the next element with accessible
      // content
      if ( !this.activeElement ) {
        this.activeElement = this.getNextElementWithAccessibleContent();
      }

      // get the accessible content for the active element, without any 'application' content
      var accessibleContent = this.getAccessibleText( this.activeElement, false ).split( ' ' );

      // if the word position is at the length of the accessible content, it is time to find the next element
      if ( this.positionInLine >= accessibleContent.length ) {
        // reset the word position
        this.positionInLine = 0;

        // update the active element and set the accessible content from this element
        this.activeElement = this.getNextElementWithAccessibleContent();
        accessibleContent = this.getAccessibleText( this.activeElement, false ).split( ' ' );
      }

      // read the next line of the accessible content
      var lineLimit = this.positionInLine + LINE_WORD_LENGTH;
      for( var i = this.positionInLine; i < lineLimit; i++ ) {
        if ( accessibleContent[ i ] ) {
          line += accessibleContent[ i ];
          this.positionInLine += 1;

          if ( accessibleContent[ i + 1 ] ) {
            line += SPACE;
          }
          else { 
            // we have reached the end of this content, there are no more words
            break;
          }
        }
      }

      this.activeLine = line;
      return line;
    },

    /**
     * Read the previous line of content from the DOM.  A line is a string of words with length
     * limitted by LINE_WORD_LENGTH;
     * 
     * @return {string}
     */
    readPreviousLine: function() {

      var line = '';

      // reset the content letter position because we have a new line
      this.letterPosition = 0;
      this.wordPosition = 0;

      // if there is no active element, set to the previous element with accessible content
      if ( !this.activeElement ) {
        this.activeElement = this.getPreviousElementWithAccessibleContent();
      }

      // get the accessible content for the active element, without any 'application' content
      var accessibleContent = this.getAccessibleText( this.activeElement, false ).split( ' ' );

      // start at the beginning of the previous line
      this.positionInLine = this.positionInLine - 2 * LINE_WORD_LENGTH;

      // if there is no content at the word position, find the previous element and start at the beginning
      if ( !accessibleContent[ this.positionInLine ] ) {
        // reset the word position
        this.positionInLine = 0;

        // update the active element and set the accessible content from this element
        this.activeElement = this.getPreviousElementWithAccessibleContent();
        accessibleContent = this.getAccessibleText( this.activeElement, false ).split( ' ' );
      }

      // read this line of content
      var lineLimit = this.positionInLine + LINE_WORD_LENGTH;
      for( var i = this.positionInLine; i < lineLimit; i++ ) {
        if ( accessibleContent[ i ] ) {
          line += accessibleContent[ i ];
          this.positionInLine += 1;

          if ( accessibleContent[ i + 1 ] ) {
            line += SPACE;
          }
          else { 
            // we have reached the end of this content, there are no more words
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
     * @return {[type]} [description]
     */
    readActiveLine: function() {

      var line = '';

      // if there is no active line, find the next one
      if ( !this.activeLine ) {
        this.activeLine = this.readNextLine();
      }

      // split up the active line into an array of words
      var activeWords = this.activeLine.split( ' ' );


      // read this line of content
      for( var i = 0; i < LINE_WORD_LENGTH; i++ ) {
        if ( activeWords[ i ] ) {
          line += activeWords[ i ];

          if ( activeWords[ i + 1 ] ) {
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

    /**
     * Read the next word in the active line.  Read the first word in the next line if we are at the end
     * of the active line.
     * 
     * @return {string}
     */
    readNextWord: function() {
      // if there is no active line, find the next one
      if ( !this.activeLine ) {
        this.activeLine = this.readNextLine();
      }

      // split the active line into an array of words
      var activeWords = this.activeLine.split( ' ' );

      // if the we are at the end of the active line, read the next one
      if ( this.wordPosition === activeWords.length ) {
        this.activeLine = this.readNextLine();
      }

      // get the word to read and increment the word position
      var outputText = activeWords[ this.wordPosition ];
      this.wordPosition++;

      return outputText;
    },

    /**
     * Read the previous word in the active line.  Read the previous line if we are at the beginning of the line.
     * 
     * @return {string}
     */
    readPreviousWord: function() {
      // if there is no active line, find the previous one
      if ( !this.activeLine ) {
        this.activeLine = this.readPreviousLine();
      }

      // if we are at the beginning of the line, read the previous one
      if ( this.wordPosition === 0 ) {
        this.activeLine = this.readPreviousLine();

        // the active word position should be at the end of the new line
        this.wordPosition = this.activeLine.split( ' ' ).length - 1;      
      }

      // split the active line into an array of words
      var activeWords = this.activeLine.split( ' ' );

      var outputText = activeWords[ this.wordPosition - 2 ];
      this.wordPosition--;

      return outputText;
    },

    /**
     * Read the next heading in the DOM of the level specified in headingLevels, 
     * relative to the position of the active element.
     *
     * @param {Array<string>} headingLevels - array of heading levels to look for
     * @return {string}
     */
    readNextHeading: function( headingLevels ) {

      // get the next element in the DOM with one of the above tag names
      var nextElement = this.getNextElementWithTagName( headingLevels );

      if ( !nextElement ) {
        // set the active element to the next element in the DOM to avoid skipping
        // the last element on backwards traversal
        this.activeElement = this.getNextElement();

        // let the user know that there are no more headings at the desired level
        if ( headingLevels.length === 1 ) {
          var noNextHeadingString = 'No next heading at ';

          var headingLevel = headingLevels[ 0 ];
          var levelString = headingLevel === 'H1' ? 'Level 1' :
                            headingLevel === 'H2' ? 'Level 2' :
                            headingLevel === 'H3' ? 'Level 3' :
                            headingLevel === 'H4' ? 'Level 4' :
                            headingLevel === 'H5' ? 'Level 5' :
                            'Level 6';


          return noNextHeadingString + levelString;
        }

        // otherwise just let the user know that there are no more headings
        return 'No more headings';
      }
      return this.getAccessibleText( nextElement );
    },

    /**
     * Read the previous heading in the parallel DOM, relative to the current heading.
     *
     * @param {Array<string>} headingLevels
     * @return {string}
     * @private
     */
    readPreviousHeading: function( headingLevels ) {

      // get the next element in the DOM with one of the above tag names
      var previousElement = this.getPreviousElementWithTagName( headingLevels );

      if ( !previousElement ) {
        // set the active element to the next element in the DOM to avoid skipping
        // the last element on backwards traversal
        this.activeElement = this.getPreviousElement();

        // let the user know that there are no more headings at the desired level
        if ( headingLevels.length === 1 ) {
          var noNextHeadingString = 'No previous heading at ';

          var headingLevel = headingLevels[ 0 ];
          var levelString = headingLevel === 'H1' ? 'Level 1' :
                            headingLevel === 'H2' ? 'Level 2' :
                            headingLevel === 'H3' ? 'Level 3' :
                            headingLevel === 'H4' ? 'Level 4' :
                            headingLevel === 'H5' ? 'Level 5' :
                            'Level 6';

          return noNextHeadingString + levelString;
        }
        return 'No previous headings';
      }
      return this.getAccessibleText( previousElement );
    },

    /**
     * Read the next form element, skipping elements that may be hidden from the user.
     * 
     * @return {string} [description]
     */
    readNextFormElement: function() {
      // list of all tag names that could be a form element
      // TODO: populate with more form elements!
      var tagNames = [ 'INPUT', 'BUTTON' ];

      var nextElement;
      var accessibleText;
      while ( !accessibleText ) {
        nextElement = this.getNextElementWithTagName( tagNames );
        accessibleText = this.getAccessibleText( nextElement );
      }

      if( accessibleText === END_OF_DOCUMENT ) {
        return 'No next form field';
      }

      return accessibleText;
    },

    /**
     * Read the next list in the document relative to the current active element, as well as the first list item
     * under the parent list.
     * 
     * @return {string}
     */
    readNextList: function() {

      // if the active element is a list item, skip to the last item to begin searching from there
      if ( this.activeElement && this.activeElement.tagName === 'LI' ) {
        var listChildren = this.activeElement.parentElement.children;
        this.activeElement = listChildren[ listChildren.length - 1 ];
      }

      // get the next list element in the DOM
      var nextElement = this.getNextElementWithTagName( [ 'UL', 'OL' ] );

      if ( !nextElement ) {
        // let the user know that there are no more lists and move to the next element
        this.activeElement = this.getNextElementWithAccessibleContent();
        return 'No more lists';
      }

      // get the content of the list element
      var listText = this.getAccessibleText( nextElement );

      // read the first item under the list
      var itemText = '';
      var firstItem = nextElement.children[ 0 ];
      if( firstItem ) {
        itemText = this.getAccessibleText( firstItem );
        this.activeElement = firstItem;
      }

      return listText + ', ' + itemText;
    },

    /**
     * Read the previous list in the document relative to the location of the current active element, as well
     * as the first list item under the parent list.
     * 
     * @return {string}
     */
    readPreviousList: function() {

      // if the active element is a list item, step outside to begin searching from the parent
      if ( this.activeElement && this.activeElement.tagName === 'LI' ) {
        this.activeElement = this.activeElement.parentElement;
      }

      // get the previous list element
      var previousElement = this.getPreviousElementWithTagName( [ 'UL', 'OL' ] );

      if( !previousElement ) {
        // let the user know that there are no previous lists
        this.activeElement = this.getPreviousElementWithAccessibleContent();
        return 'No previous lists';
      }

      // get the content from the list element
      var listText = this.getAccessibleText( previousElement );

      // include the content from the first item in the list
      var itemText = '';
      var firstItem = previousElement.children[ 0 ];
      if ( firstItem ) {
        itemText = this.getAccessibleText( firstItem );
        this.activeElement = firstItem;
      }

      return listText + ', ' + itemText;
    },

    /**
     * Read the next item or the next list if no list is selected.
     * 
     * @return {}
     */
    readNextListItem: function() {

      if ( !this.activeElement ) {
        this.activeElement = this.getNextElementWithAccessibleContent();
      }

      // if we are not inside of a list or we are at the end of a list, get the next list
      var listItems = this.activeElement.parentElement.children;
      if ( this.activeElement.tagName !== 'LI'  || this.activeElement === listItems[ listItems.length - 1 ] ) {
        var nextList = this.getNextElementWithTagName( [ 'OL', 'UL' ] );

        if( nextList ) {
          var itemElement = nextList.children[ 0 ];
          var listContent = this.getAccessibleText( nextList );
          var itemContent = this.getAccessibleText( itemElement );

          this.activeElement = itemElement;

          return listContent + SPACE + itemContent;
        }
        else {
          // let the user know that there are no more lists and set the active element to the next item
          this.activeElement = this.getNextElementWithAccessibleContent();
          return 'No more lists';
        }
      }
      else {
        // otherwise read content from the next peer
        var nextElement = this.activeElement.nextSibling;
        this.activeElement = nextElement;
        return this.getAccessibleText( nextElement );
      }

    },

    /**
     * Read the previous list item in the document.  If the active element is outside of a list, read the previous
     * list and its first item.
     * 
     * @return {string}
     */
    readPreviousListItem: function() {
      console.log( this.activeElement );

      if ( !this.activeElement ) {
        this.activeElement = this.getNextElementWithAccessibleContent();
      }

      // if we are not inside of a list or we are at the first list item, get the previous list
      var listItems = this.activeElement.parentElement.children;
      if ( this.activeElement.tagName !== 'LI' || this.activeElement === listItems[ 0 ] ) {

        // step out of the current list
        if ( this.activeElement === listItems[ 0 ] ) {
          this.activeElement = this.activeElement.parentElement;
        }

        var previousList = this.getPreviousElementWithTagName( [ 'OL', 'UL' ] );

        if( previousList ) {
          var itemElement = previousList.children[ 0 ];
          var listContent = this.getAccessibleText( previousList );
          var itemContent = this.getAccessibleText( itemElement );

          this.activeElement = itemElement;

          return listContent + SPACE + itemContent;
        }
        else {
          // let the user know that there are no more lists
          this.activeElement = this.getPreviousElementWithAccessibleContent();
          return 'No previous lists';
        }
      }
      else {
        // otherwise, get the previous peer
        var previousElement = this.activeElement.previousSibling;
        this.activeElement = previousElement;
        return this.getAccessibleText( previousElement );
      }
    },

    /**
     * Read the previous form element skipping elements that may be hidden from the user.
     * @return {[type]} [description]
     */
    readPreviousFormElement: function() {

      // TODO: populate with more elements!
      var tagNames = [ 'INPUT', 'BUTTON' ];

      var previousElement;
      var accessibleText;
      while ( !accessibleText ) {
        previousElement = this.getPreviousElementWithTagName( tagNames );
        accessibleText = this.getAccessibleText( previousElement );
      }

      if ( accessibleText === END_OF_DOCUMENT ) {
        this.activeElement = this.getPreviousElement();
        return 'No previous form field';
      }

      return accessibleText;
    },

    /**
     * Read the next character in the currently active line.
     * 
     * @return {string}
     * @private
     */
    readNextCharacter: function() {

      // if there is no active line, find the next one
      if ( !this.activeLine ) {
        this.activeLine = this.readNextLine();
      }

      // if the we are at the end of the active line, read the next one
      if ( this.letterPosition === this.activeLine.length ) {
        this.activeLine = this.readNextLine();
      }

      // get the letter to read and increment the letter position
      var outputText = this.activeLine[ this.letterPosition ];
      this.letterPosition++;

      return outputText;
    },

    /**
     * Read the previous character in the active line.
     * 
     * @return {string}
     */
    readPreviousCharacter: function() {

      // if there is no active line, find the previous one
      if ( !this.activeLine ) {
        this.activeLine = this.readPreviousLine();
      }

      // if we are already at the begining of the line, we should go back to the previous line
      if ( this.letterPosition === 0 ) {
        this.activeLine = this.readPreviousLine();

        // since we are moving backwards through the document, we need to set the letter position to the
        // end of the active line
        this.letterPosition = this.activeLine.length;
      }

      // get the letter to read and decrement the letter position
      var outputText = this.activeLine[ this.letterPosition - 2 ];
      this.letterPosition--;

      return outputText;
    },


    /**
     * Update the list of elements, and add Mutation Observers to each one.  MutationObservers
     * provide a way to listen to changes in the DOM,
     * see https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver
     */
    updateLiveElementList: function() {

      var thisCursor = this;

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
        var liveRole = thisCursor.getLiveRole( domElement );

        if( liveRole ) {
          var mutationObserverCallback = function( mutations ) {
            mutations.forEach( function( mutation ) {
              var liveRole;
              var mutatedElement = mutation.target;

              // look for the type of live role that is associated with this mutation
              // if the target has no live attribute, search through the element's ancestors to find the attribute
              while( !liveRole ) {
                liveRole = thisCursor.getLiveRole( mutatedElement );
                mutatedElement = mutatedElement.parentElement;
              }

              var updatedText = mutation.addedNodes[ 0 ].data;
              thisCursor.outputUtteranceProperty.set( new Utterance( updatedText, liveRole ) );
            } );
          };

          // create a mutation observer for this live element
          var observer = new MutationObserver( function( mutations ) {
            mutationObserverCallback( mutations );
          } );

          // listen for changes to the subtree in case children of the aria-live parent change their textContent
          var observerConfig = { childList: true, subtree: true };

          observer.observe( domElement, observerConfig );
          thisCursor.observers.push( observer );
        }
      }
    },

    /**
     * Read continuously from the current active element.  Accessible content is read by reader with a 'polite' 
     * utterance so that new text is added to the queue line by line.
     *
     * TODO: If the read is cancelled, the active element should be set appropriately.
     * 
     * @return {string}
     */
    readEntireDocument: function() {

      var liveRole = 'polite';
      var outputText = this.getAccessibleText( this.activeElement );
      var activeElement = this.activeElement;

      while ( outputText !== END_OF_DOCUMENT ) {
        activeElement = this.activeElement;
        outputText = this.readNextLine();

        if ( outputText === END_OF_DOCUMENT ) {
          this.activeElement = activeElement;
        }
        // var nextElement = this.getNextElementWithAccessibleContent();
        // outputText = this.getAccessibleText( nextElement );

        this.outputUtteranceProperty.set( new Utterance( outputText, liveRole ) );
      }
    },

    /**
     * Return true if the element is focusable.  A focusable element has a tab index, or is a 
     * form element.
     *
     * TODO: Populate with the rest of the focusable elements.
     * @param  {DOMElement} domElement
     * @return {Boolean}
     */
    isFocusable: function( domElement ) {
      if ( domElement.getAttribute( 'tabindex' ) || domElement.tagName === 'BUTTON' ) {
        return true;
      }
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