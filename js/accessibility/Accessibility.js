// Copyright 2017, University of Colorado Boulder

/**
 * A mixin for Node that implements accessibility features adding accessible HTML content to the parallel DOM.
 *
 * The parallel DOM is an HTML structure that provides semantics for assistive technologies. For web content to be
 * accessible, assistive technologies require HTML markup, which is something that pure graphical content does not
 * include.  AccessibleNode implements the accessible HTML content for a node in the scene graph.  The parallel DOM
 * contains accessible content for each AccessibleNode in the scene graph.
 *
 * Each node can have accessible content.  The structure of the accessible content will match the structure of the scene
 * graph.
 *
 * Say we have the following scene graph:
 *
 *   A
 *  / \
 * B   C
 *    / \
 *   D   E
 *        \
 *         F
 *
 * And say that nodes A, B, C, D, and F specify accessible content for the DOM.  Scenery will render the accessible
 * content like so:
 *
 * <div id="node-A">
 *   <div id="node-B"></div>
 *   <div id="node-C">
 *     <div id="node-D"></div>
 *     <div id="node-F"></div>
 *   </div>
 * </div>
 *
 * In this example, each element is represented by a div, but any HTML element could be used. Note that in this example,
 * node E did not specify accessible content, so node F was added as a child under node C.  If node E had specified
 * accessible content, content for node F would have been added as a child under the content for node E.
 *
 * It is possible to add additional structure to the accessible content if necessary.  For instance, consider the
 * following accessible content for a button node:
 *
 * <div>
 *   <button>Button label</button>
 *   <p>This is a description for the button</p>
 * </div>
 *
 * The node is represented by the <button> DOM element, but the accessible content needs to include the parent div, and
 * a peer description paragraph.  AccessibleNode supports this structure with the 'parentContainerElement' option.  In
 * this example, the parentContainerElement is the div, while the description is added as a child under the button
 * node's domElement.
 *
 * There is also a `childContainerElement`, which is a container element for accessible content for children under this
 * node.  For instance, consider the following scene graph:
 *
 *   A
 *  / \
 * B   C
 *
 * And suppose you want add a specific label and description to node A without altering the structure of the scene graph
 * with HTML content like this:
 *
 * <div id="node-A">
 *   <p>Label for node A</p>
 *   <p>Label for node B</p>
 *
 *   <div id='childContainerElement'>
 *     <div id="node-B"></div>
 *     <div id="node-C"></div>
 *   </div>
 * </div>
 *
 * The above is easily handled by passing in an optional 'childContainerTagName', which will create a child container
 * element with the desired tag name.
 *
 * For additional accessibility options, please see the options at the top of the AccessibleNode constructor. For more
 * documentation on Scenery, Nodes, and the scene graph, please see http://phetsims.github.io/scenery/
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );
  // we use the following but comment out to avoid circular dependency
  // var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );

  // constants
  // for specifying direction of DOM traversal
  var NEXT = 'NEXT';
  var PREVIOUS = 'PREVIOUS';

  // specific DOM tag names, used frequently and for validation
  var DOM_INPUT = 'INPUT';
  var DOM_LABEL = 'LABEL';
  var DOM_UNORDERED_LIST = 'UL';
  var DOM_BUTTON = 'BUTTON';
  var DOM_TEXTAREA = 'TEXTAREA';
  var DOM_SELECT = 'SELECT';
  var DOM_OPTGROUP = 'OPTGROUP';
  var DOM_DATALIST = 'DATALIST';
  var DOM_OUTPUT = 'OUTPUT';

  // these elements will have labels that use inner text
  var ELEMENTS_WITH_INNER_TEXT = [ DOM_BUTTON ];

  // these elements are typically associated with forms, and support certain attributes
  var FORM_ELEMENTS = [ DOM_INPUT, DOM_BUTTON, DOM_TEXTAREA, DOM_SELECT, DOM_OPTGROUP, DOM_DATALIST, DOM_OUTPUT ];

  // global incremented to provide unique id's
  var ITEM_NUMBER = 0;

  var Accessibility = {

    /**
     * Given the constructor for Node, mix accessibility functions into the prototype
     * 
     * @param  {function} type - the constructor for Node
     */
    mixin: function( type  ) {
      var proto = type.prototype;

      //----------------------------------------------------------------------------------------------------------------
      // add accessibility functions to the prototype
      //----------------------------------------------------------------------------------------------------------------

      /**
       * Mix accessibility into an instance of a node, instantiating the accessible peer and creating 
       * representative DOM elements.
       * 
       * @param  {Object} options
       * @public (scenery-internal)
       */
      proto.mixinAccessibility = function( options ) {
        options = _.extend( {

          // html tag names
          tagName: null, // {string} - tag name for the element representing this node in the parallel DOM
          inputType: null, // {string} - specify the input type attribute - only relevant if tagName is 'input'
          parentContainerTagName: null, // {string} - creates a parent container DOM element for this node's DOM element and its peers
          childContainerTagName: null, // {string} - creates a container element for the children under this node's DOM element
          labelTagName: null, // {string} - tag name for the element containing the label, usually a paragraph, label, or heading
          descriptionTagName: null, // - tag name for the element containing the description, usually a paragraph or a list item

          // focus highlight
          focusHighlight: null, // {Node|Shape|Bounds2} - default is a pink rectangle around the node's local bounds

          // accessible labels
          label: null, // {string} - accessible label applied to this node, usually read by the screen reader on focus
          useAriaLabel: false, // {boolean} - if true, a label element will not be created and the label will be inline with aria-label

          // accessible descriptions
          description: '', // {string} - description for this node in the parallel DOM, only read on focus if ariaDescribedBy defined

          // DOM events
          events: {}, // {object} - object with keys of type event name, values of type function callback for the event

          // attributes
          hidden: false, // {boolean} - hides the element in the parallel DOM from the AT
          focusable: false, // {boolean} = whether or not the element can receive keyboard focus

          // aria
          attributes: [], // {Object[]} - objects specifying element attributes - keys attribute and value of attribute value
          ariaRole: null, // {string} - aria role for the element, can define extra semantics for the reader, see https://www.w3.org/TR/wai-aria/roles for list of roles
          ariaDescribedBy: null, // {string} - an ID of a description element to describe this dom element
          ariaLabelledBy: null // {string} - an ID of a label element to describe this dom element
        }, options );

        //----------------------------------------------------------------------------------------------------------------
        // Instantiate the accessible content
        //----------------------------------------------------------------------------------------------------------------
        
        // no need to create accessible content if there is no tagname
        if ( !options.tagName ) {
          return;
        }

        // @private - the dom element representing this node in the parallel DOM
        this._domElement = this.createDOMElement( options.tagName );

        // @private - whether or not to use the aria-label attribute
        this._useAriaLabel = options.useAriaLabel;

        // @private - the description for this dom element
        this._descriptionElement = null;
        this._description = options.description;
        options.descriptionTagName && this.createDescriptionElement( options.descriptionTagName, 'description' );

        // @private - the label for this dom element
        // the label might be an element, or it could be inline text on the dom element
        this._labelElement = null;
        this._label = options.label;
        this._labelTagName = options.labelTagName;
        this.addLabel();

        // @private - set tab index if explicitly added or removed from navigation order
        this._focusable = options.focusable;
        this.setFocusable( this._focusable );

        // @private
        this._hidden = options.hidden;
        this._hidden && this.setHidden( this._hidden );

        // @private
        this._ariaRole = options.ariaRole;

        // add aria role
        options.ariaRole && this.setAttribute( 'role', options.ariaRole );

        // add input type if supported and defined
        options.inputType && this.setInputType( options.inputType );

        // add additional attributes
        options.attributes.forEach( function( attribute ) {
          this.setAttribute( attribute.attribute, attribute.value );
        } );

        // @private - parent container for this node's dom element and its peers
        this._parentContainerElement = null;

        // @private - container element for children under this node's dom element
        this._childContainerElement = null;

        if ( options.parentContainerTagName ) {
          this._parentContainerElement = this.createDOMElement( options.parentContainerTagName, 'parent-container-' + this.id );

          // if using a parent container, it should contain the description and label as peers of this dom element
          this._labelElement && this._parentContainerElement.appendChild( this._labelElement );
          this._descriptionElement && this._parentContainerElement.appendChild( this._descriptionElement );
        }
        else if ( options.childContainerTagName ) {

          // if using a child container element, label and description come first
          this._labelElement && this._domElement.appendChild( this._labelElement );
          this._descriptionElement && this._domElement.appendChild( this._descriptionElement );

          this._childContainerElement = this.createDOMElement( options.childContainerTagName, 'child-container-' + this.id );
          this._domElement.appendChild( this._childContainerElement );
        }
        else {
          this._labelElement && this._domElement.appendChild( this._labelElement );
          this._descriptionElement && this._domElement.appendChild( this._descriptionElement );
        }

        // now set the accessible content by creating an accessible peer, assuming that
        // a dom element was defined
        var self = this;
        this.accessibleContent = {
          focusHighlight: options.focusHighlight,
          createPeer: function( accessibleInstance ) {

            // register listeners to the events
            for ( var event in options.events ) {
              if ( options.events.hasOwnProperty( event ) ) {
                self.addDOMEventListener( event, options.events[ event ] );
              }
            }

            self.childContainerElement && self._domElement.appendChild( self.childContainerElement );

            // add an aria-describedby attribute if it is specified in options
            options.ariaDescribedBy && self.setAriaDescribedBy( options.ariaDescribedBy );

            // add an aria-labelledby attribute if it is specified in options
            options.ariaLabelledBy && self.setAriaLabelledBy( options.ariaLabelledBy );

            return new scenery.AccessiblePeer( accessibleInstance, self._domElement, {
              parentContainerElement: self._parentContainerElement,
              childContainerElement: self._childContainerElement
            } );
          }
        };
      };

      /**
       * Create an element to represent this node in the parallel DOM.
       *
       * @private
       * @param  {string} tagName
       * @param {string} [id] - optional id for the dom element, will be the node's id if not defined
       */
      proto.createDOMElement = function( tagName, id ) {
        var domElement = document.createElement( tagName );
        domElement.id = id || this.id;
        return domElement;
      };

      /**
       * Add the label for this element.  The label can be added in one of four ways.
       *  - As inline text with the 'aria-label' attribute.
       *  - As a 'label' element with the 'for' attribute pointing to this node's dom element
       *  - As inner text on the node's dom element itself
       *  - As a separate dom element positioned as a peer or child of this nod'es dom element
       *
       * @private
       */
      proto.addLabel = function() {
        if ( this._useAriaLabel ) {
          this.setAttribute( 'aria-label', this._label );
        }
        else if ( this.elementSupportsInnerText() ) {
          this.domElement.innerText = this._label;
        }
        else if ( this._labelTagName ) {

          // the remaining methods require a new DOM element
          this._labelElement = this.createDOMElement( this._labelTagName, 'label-' + this.id  );
          this._labelElement.textContent = this._label;

          if ( this._labelTagName.toUpperCase() === DOM_LABEL ) {
            this._labelElement.setAttribute( 'for', this.id );
          }
        }
      };

      /**
       * Create the description element for this node's dom element. If a description
       * was passed through options, set it immediately.
       *
       * @param {string} tagName
       * @private
       */
      proto.createDescriptionElement = function( tagName ) {
        this._descriptionElement = this.createDOMElement( tagName, 'description-' + this.id );
        this._description && this.setDescription( this._description );
      };

      /**
       * Set the input type.  Assert that the tagname is input.
       * @param {string} type
       * @private
       */
      proto.setInputType = function( type ) {
        assert && assert( this._domElement.tagName === DOM_INPUT, 'input type can only be set on element with tagname INPUT' );
        this.setAttribute( 'type', type );
      };

      /**
       * Some types support inner text, and these types should have a label
       * defined this way, rather than a second paragraph contained in a parent element.
       *
       * @return {boolean}
       * @private
       */
      proto.elementSupportsInnerText = function() {
        var supportsInnerText = false;

        // more input types will need to be added here
        for ( var i = 0; i < ELEMENTS_WITH_INNER_TEXT.length; i++ ) {
          if ( this._domElement.tagName === ELEMENTS_WITH_INNER_TEXT[ i ] ) {
            supportsInnerText = true;
          }
        }
        return supportsInnerText;
      };

      /**
       * Get the dom element representing this node.
       * @public
       * @return {HTMLElement}
       */
      proto.getDOMElement = function() {
        return this._domElement;
      };
      Object.defineProperty( proto, 'domElement', { get: function() { return this.getDOMElement(); } } );

      /**
       * Get the ARIA role representing this node.
       * @public
       * @return {string}
       */
      proto.getAriaRole = function() {
        return this._ariaRole;
      };
      Object.defineProperty( proto, 'ariaRole', { get: function() { return this.getAriaRole(); } } );

      /**
       * Set the focus highlight for this node.
       * 
       * @param {Node|Shape|string.<'invisible'>} focusHighlight
       * @public
       */
      proto.setFocusHighlight = function( focusHighlight ) {
        var accessibleContent = this.accessibleContent;
        accessibleContent.focusHighlight = focusHighlight;

        this.setAccessibleContent( accessibleContent );
      };
      Object.defineProperty( proto, 'focusHighlight', { set: function( focusHighlight ) { this.setFocusHighlight( focusHighlight ); } } );

      /**
       * Set the text content for the label element of this node.  The label element
       * is usually either a paragraph, a label, or innerText for a certain inputs.
       *
       * @param  {string} textContent
       * @public
       */
      proto.setLabel = function( textContent ) {
        if ( !this.elementSupportsInnerText() ) {
          this._labelElement.textContent = textContent;
        }
        else if ( this._useAriaLabel ) {
          this.setAttribute( 'aria-label', textContent );
        }
        else {
          this._domElement.textContent = textContent;
        }
      };

      /**
       * Set the description content for this node.  If the node is described by a list, please
       * update description list items individually with updateDescriptionItem.
       *
       * @param {string} textContent
       * @public
       */
      proto.setDescription = function( textContent ) {
        assert && assert( this._descriptionElement, 'desription element must exist in prallel DOM' );
        assert && assert( this._descriptionElement.tagName !== DOM_UNORDERED_LIST, 'list description in use, please use  ' );
        this._descriptionElement.textContent = textContent;
      };

      /**
       * Get an id referencing the description element of this node.  Useful when you want to
       * set aria-describedby on a DOM element that is far from this one in the scene graph.
       *
       * @return {string}
       * @public
       */
      proto.getDescriptionElementID = function() {
        assert && assert( this._descriptionElement, 'description element must exist in the parallel DOM' );
        return this._descriptionElement.id;
      };

      /**
       * Get an id referencing the label element of this node.  Useful when you want to
       * set aria-labelledby on a DOM element that is far from this one in the scene graph.
       *
       * @return {string}
       * @public
       */
      proto.getLabelElementID = function() {
        assert && assert( this._labelElement, 'description element must exist in the parallel DOM' );
        return this._labelElement.id;
      };

      /**
       * Add the 'aria-describedby' attribute to this node's dom element.
       *
       * @param {string} [descriptionID] - optional id referencing the description element
       * @public
       */
      proto.setAriaDescribedBy = function( descriptionID ) {
        this._domElement.setAttribute( 'aria-describedby', descriptionID );
      };

      /**
       * Add the 'aria-labelledby' attribute to this node's dom element.
       *
       * @param {string} [labelID] - optional id referencing the description element
       * @public
       */
      proto.setAriaLabelledBy = function( labelID ) {
        this._domElement.setAttribute( 'aria-labelledby', labelID );
      };

      /**
       * If the node is using a list for its description, add a list item to the end of the list with
       * the text content.  Returns an id so that the element can be referenced if need be.
       *
       * @param  {string} textContent
       * @return {string}
       * @public
       */
      proto.addDescriptionItem = function( textContent ) {
        assert && assert( this._descriptionElement.tagName === DOM_UNORDERED_LIST, 'description element must be a list to use addDescriptionItem' );

        var listItem = document.createElement( 'li' );
        listItem.textContent = textContent;
        listItem.id = 'list-item-' + ITEM_NUMBER++;
        this._descriptionElement.appendChild( listItem );

        return listItem.id;
      };

      /**
       * Update the text content of the description item.  The item may not yet be in the DOM, so
       * document.getElementById cannot be used, and the element needs to be found
       * under the description element.
       *
       * @param  {string} itemID - id of the lits item to update
       * @param  {string} description - new textContent for the string
       * @public
       */
      proto.updateDescriptionItem = function( itemID, description ) {
        var listItem = this.getChildElementWithId( this._descriptionElement, itemID );
        assert && assert( this._descriptionElement.tagName === DOM_UNORDERED_LIST, 'description must be a list to hide list items' );
        assert && assert( listItem, 'No list item in description with id ' + itemID );

        listItem.textContent = description;
      };

      /**
       * Hide or show the desired list item from the screen reader
       *
       * @param {string} itemID - id of the list item to hide
       * @param {boolean} hidden - whether the list item should be hidden
       * @public
       */
      proto.setDescriptionItemHidden = function( itemID, hidden ) {
        var listItem = document.getElementById( itemID );
        assert && assert( this._descriptionElement.tagName === DOM_UNORDERED_LIST, 'description must be a list to hide list items' );
        assert && assert( listItem, 'No list item in description with id ' + itemID );

        listItem.hidden = hidden;
      };

      /**
       * Hide completely from a screen reader by setting the aria-hidden attribute. If this domElement
       * and its peers have a parent container, it should be hidden.
       *
       * @param {boolean} hidden
       * @public
       */
      proto.setHidden = function( hidden ) {
        if ( this._parentContainerElement ) {
          this._parentContainerElement.hidden = hidden;
        }
        else {
          this._domElement.hidden = hidden;
        }
        this._hidden = hidden;
      };
      Object.defineProperty( proto, 'hidden', { set: function( hidden ) { this.setHidden( hidden ); } } );

      /**
       * Set the value of an input element.
       * 
       * @param {string} value [description]
       */
      proto.setInputValue = function( value ) {
        assert && assert( _.contains( FORM_ELEMENTS, this._domElement.tagName ), 'dom element must be a form element to support value' );

        this._domElement.value = value;
      };

      /**
       * Get the value of the element.
       * 
       * @return {string}
       */
      proto.getInputValue = function() {
        assert && assert( _.contains( FORM_ELEMENTS, this._domElement.tagName ), 'dom element must be a form element to support value' );

        return this._domElement.value;
      };
      Object.defineProperty( proto, 'inputValue', {
        set: function( inputValue ) { this.setInputValue( inputValue ); },
        get: function() { this.getInputValue(); }
      } );

      /**
       * Set a particular attribute for this node's dom element, generally to provide extra
       * semantic information for a screen reader.
       *
       * @param  {string} attribute - string naming the attribute
       * @param  {string|boolean} value - the value for the attribute
       * @public
       */
      proto.setAttribute = function( attribute, value ) {
        this._domElement.setAttribute( attribute, value );
      };

      /**
       * Remove a particular attribute, removing the associated semantic information from
       * the DOM element.
       *
       * @param  {string} attribute - name of the attribute to remove
       * @public
       */
      proto.removeAttribute = function( attribute ) {
        this._domElement.removeAttribute( attribute );
      };

      /**
       * Make the container dom element focusable.
       *
       * @param {boolean} isFocusable
       * @public
       */
      proto.setFocusable = function( isFocusable ) {
        this._focusable = isFocusable;
        this._domElement.tabIndex = isFocusable ? 0 : -1;
      };

      /**
       * Get if this node is focusable by a keyboard.
       *
       * @return {boolean}
       * @public
       */
      proto.getFocusable = function() {
        return this._focusable;
      };
      Object.defineProperty( proto, 'focusable', {
        set: function( focusable ) { this.setFocusable( focusable ); },
        get: function() { return this.getFocusable(); }
      } );

      /**
       * Focus this node's dom element.
       *
       * @public
       */
      proto.focus = function() {
        assert && assert( this._focusable, 'trying to set focus on a node that is not focusable' );

        // make sure that the elememnt is in the navigation order
        this.setFocusable( true );
        this._domElement.focus();
      };

      /**
       * Add an event listener to the dom element.  The listener is usually meant to keep
       * the attributes of the element up to date with properties of the sim.
       * 
       * @param {string} eventName - name of the event to listen to
       * @param {function} callBack
       */
      proto.addDOMEventListener = function( eventName, callBack ) {
        this._domElement.addEventListener( eventName, callBack );
      };

      /**
       * Remove an event listener from the DOM element, when you want to stop listening
       * or dispose.
       * 
       * @param  {string} eventName
       * @param  {function} callBack
       */
      proto.removeDOMEventListener = function( eventName, callBack ) {
        this._domElement.removeEventListener( eventName, callBack );
      };

      /**
       * Get all 'element' nodes off the parent element, placing them in an array
       * for easy traversal.  Note that this includes all elements, even those
       * that are 'hidden' or purely for structure.
       *
       * @param  {HTMLElement} domElement - the parent element to linearize
       * @return {HTMLElement[]}
       * @private
       */
      proto.getLinearDOMElements = function( domElement ) {

        // gets ALL descendant children for the element
        var children = domElement.getElementsByTagName( '*' );

        var linearDOM = [];
        for ( var i = 0; i < children.length; i++ ) {

          // searching for the HTML elemetn nodes
          if ( children[ i ].nodeType === HTMLElement.ELEMENT_NODE ) {
            linearDOM[ i ] = ( children[ i ] );
          }
        }
        return linearDOM;
      };

      /**
       * Get the next focusable element in the parallel DOM.
       *
       * @return {HTMLElement}
       * @public
       */
      proto.getNextFocusable = function() {
        return this.getNextPreviousFocusable( NEXT );
      };

      /**
       * Get the previous focusable elements in the parallel DOM
       *
       * @return {HTMLElement}
       * @public
       */
      proto.getPreviousFocusable = function() {
        return this.getNextPreviousFocusable( PREVIOUS );
      };

      /**
       * Make eligible for garbage collection.
       *
       * @public
       */
      proto.dispose = function() {
        this.disposeAccessibleNode();
      };

      /**
       * Get the next or previous focusable element in the parallel DOM, relative to this Node's domElement 
       * depending on the parameter.  Useful if you need to set focus dynamically or need to prevent
       * default behavior for the tab key.
       *
       * @param {string} direction - direction of traversal, one of 'Next' | 'PREVIOUS'
       * @return {HTMLElement}
       * @private
       */
      proto.getNextPreviousFocusable = function( direction ) {

        // linearize the DOM or easy traversal
        var linearDOM = this.getLinearDOMElements( document.getElementsByClassName( 'accessibility' )[ 0 ] );

        var activeElement = this._domElement;
        var activeIndex = linearDOM.indexOf( activeElement );
        var delta = direction === NEXT ? +1 : -1;

        // find the next focusable element in the DOM
        var nextIndex = activeIndex + delta;
        var nextFocusable;
        while ( !nextFocusable && nextIndex < linearDOM.length - 1 ) {
          var nextElement = linearDOM[ nextIndex ];
          nextIndex += delta;

          // continue to while if the next element is meant to be hidden
          if ( nextElement.hidden ) {
            continue;
          }

          if ( nextElement.tabIndex > -1 ) {
            nextFocusable = nextElement;
            break;
          }
        }

        // if no next focusable is found, return this DOMElement
        return nextFocusable || this._domElement;
      };

      /**
       * Get a child element with an id.  This should only be used if the element is not in the document.
       * If the element is in the document, document.getElementById is a faster (and more conventional)
       * option.
       *
       * This function is still useful because elements can exist before being added to the DOM during
       * instantiation of the Node's peer.
       *
       * @param  {HTMLElement} parentElement
       * @param  {string} childId
       * @return {HTMLElement}
       */
      proto.getChildElementWithId = function( parentElement, childId ) {
        var childElement;
        var children = parentElement.children;

        for ( var i = 0; i < children.length; i++ ) {
          if ( children[ i ].id === childId ) {
            childElement = children[ i ];
            break;
          }
        }

        if ( !childElement ) {
          throw new Error( 'No child element under ' + parentElement + ' with id ' + childId );
        }

        return childElement;
      };
    }
  };

  scenery.register( 'Accessibility', Accessibility );

  return Accessibility;
} );