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
  var extend = require( 'PHET_CORE/extend' );
  var AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );
  // we use the following but comment out to avoid circular dependency
  // var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );

  // constants
  // global incremented to provide unique id's
  var ITEM_NUMBER = 0;

  var ACCESSIBILITY_OPTION_KEYS = [
    'tagName', // Sets the tag name for the DOM element representing this node in the parallel DOM
    'inputType', // Sets the input type for the representative DOM element, only relevant if tagname is 'input'
    'parentContainerTagName', // Sets the tag name for an element that contains this node's DOM element and its peers
    'labelTagName', // Sets the tag name for the DOM element labelling this node, usually a paragraph
    'descriptionTagName', // Sets the tag name for the DOM element describing this node, usually a paragraph
    'focusHighlight', // Sets the focus highlight for the node, see setFocusHighlight()
    'accessibleLabel', // Set the label content for the node
    'accessibleDescription', // Set the description content for the node
    'accessibleHidden', // Sets wheter or not the node's DOM element is hidden in the parallel DOM
    'focusable', // Sets whether or not the node can receive keyboard focus
    'useAriaLabel', // Sets whether or not the label will use the 'aria-label' attribute
    'ariaRole', // Sets the ARIA role for the DOM element, see setAriaRole() for documentation
    'ariaDescribedById', // Sets a description relationship for this node's DOM element by id, see setAriaDescribedById()
    'ariaLabelledById', // Sets a label relationship with another element in the DOM by id, see setAriaLabelledById()
    'prependLabels'// Sets whether we want to prepend labels above the node's HTML element, see setPrependLabels()
  ];

  var Accessibility = {

    /**
     * Given the constructor for Node, mix accessibility functions into the prototype
     * 
     * @param {function} type - the constructor for Node
     */
    mixin: function( type  ) {
      var proto = type.prototype;

      /**
       * These properties and methods are put directly on the prototype of nodes
       * that have Accessibility mixed in.
       */
      extend( proto, {

        /**
         * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
         * order they will be evaluated in.  Order does not matter for accessibility options, though order does matter
         * for Scenery's Node.js and its other mixins.
         * @protected
         *
         * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
         *       cases that may apply.
         */
        _mutatorKeys: ACCESSIBILITY_OPTION_KEYS.concat( proto._mutatorKeys ),

        /**
         * This should be called in the constructor to initialize the accessibility-specific
         * parts of Node.
         * @protected
         */
        initializeAccessibility: function() {

          // tag names
          this._tagName = null;
          this._inputTagName = null;
          this._parentContainerTagName = null;
          this._labelTagName = null;
          this._descriptionTagName = null;

          // DOM elements
          this._domElement = null;
          this._labelElement = null;
          this._descriptionElement = null;
          this._parentContainerElement = null;

          // array of attributes on the DOM Element.  The array will contain objects
          // of the form { attribute: {string}, value: {string|number} } 
          this._accessibleAttributes = [];

          // content
          this._accessibleLabel = null;
          this._accessibleDescription = null;
          this._prependLabels = null;

          // aria
          this._useAriaLabel = null;
          this._ariaRole = null;
          this._ariaDescribedById = null;
          this._ariaLabelledById = null;

          this._focusable = null;
          this._focusHighlight = null;
          this._accessibleHidden = null;

          // An id used for the purposes of accessibility.  The accessible id
          // is a string, since the DOM API frequently uses string identifiers
          // to reference other elements in the DOM.
          this._accessibleId = AccessibilityUtil.generateHTMLElementId();

          // @private {Array.<Function>} - For accessibility input handling {keyboard/click/HTML form}
          this._accessibleInputListeners = [];

        },

        /**
         * Create a DOM element, giving it a unique ID if necessary.
         * @private
         * 
         * @param  {string} tagName
         * @param {string} [id] - optional id for the dom element, will be the node's id if not defined
         */
        createDOMElement: function( tagName ) {
          return document.createElement( tagName );
        },

        /**
         * Add DOM event listeners contained in the accessibleInput directly
         * to the DOM element.
         * @private
         * 
         * @param {Object} accessibleInput
         */
        addDOMEventListeners: function( accessibleInput ) {
          var self = this;
          for ( var event in accessibleInput ) {
            if ( accessibleInput.hasOwnProperty( event ) ) {
              self._domElement.addEventListener( event, accessibleInput[ event ] );
            }
          }
        },

        /**
         * Remove a DOM event listener contained in an accesssibleInput.
         * @private
         *
         * @param  {Object} accesssibleInput
         */
        removeDOMEventListeners: function( accesssibleInput ) {
          var self = this;
          for ( var event in accesssibleInput ) {
            if ( accesssibleInput.hasOwnProperty( event ) ) {
              self._domElement.removeEventListener( event, accesssibleInput[ event ] );
            }
          }
        },

        /**
         * Adds an accessibility input listener.
         * @public
         *
         * @param {Object} listener
         * @returns {Node} - Returns 'this' reference, for chaining
         */
        addAccessibleInputListener: function( accessibleInput ) {
          // don't allow listeners to be added multiple times
          if ( _.indexOf( this._accessibleInputListeners, accessibleInput ) === -1 ) {
            this._accessibleInputListeners.push( accessibleInput );

            // add the event listener to the DOM element
            this.addDOMEventListeners( accessibleInput );
          }
          return this;
        },

        /**
         * Removes an input listener that was previously added with addAccessibleInputListener.
         * @public
         *
         * @param {Object} listener
         * @returns {Node} - Returns 'this' reference, for chaining
         */
        removeAccessibleInputListener: function( listener ) {
          // ensure the listener is in our list
          assert && assert( _.indexOf( this._accessibleInputListeners, listener ) !== -1 );
          this._accessibleInputListeners.splice( _.indexOf( this._accessibleInputListeners, listener ), 1 );

          // remove the event listeners from the DOM element
          this.removeDOMEventListeners( listener );

          return this;
        },

        /**
         * Returns a copy of all accessibiltiy input listeners.
         * @public
         *
         * @returns {Array.<Object>}
         */
        getAccessibleInputListeners: function() {
          return this._accessibleInputListeners.slice( 0 ); // defensive copy
        },
        get accessibleInputListeners() { return this.getAccessibleInputListeners(); },

        /**
         * Get the accesible id for this node.  The accessible id is built from
         * the node's trail in invalidateAccessibleContent. It is
         * a string since elemetns are are generally referenced by string in the
         * the DOM API.
         * 
         * @return {string}
         */
        getAccessibleId: function() {
          return this._accessibleId;
        },
        get accessibleId() { return this.getAccessibleId(); },

        /**
         * Set the tag name representing this element in the DOM. DOM element 
         * tag names are read-only, so this function will create a new DOM
         * element for the Node and reset the accessible content.
         *  
         * @param {string} tagName
         */
        setTagName: function( tagName ) {
          this._tagName = tagName;
          this._domElement = this.createDOMElement( tagName );
          this._domElement.id = this._accessibleId;

          // Safari seems to require that a range input has a width, otherwise it will not be keyboard accessible.
          if ( _.contains( AccessibilityUtil.ELEMENTS_THAT_NEED_WIDTH, tagName ) ) {
            this._domElement.style.width = '1px';
          }

          this.invalidateAccessibleContent();
        },
        set tagName( tagName ) { this.setTagName( tagName ); },

        /**
         * Get the tag name of the DOM element representing this node for 
         * accessibility.
         * @public
         * 
         * @return {string}
         */
        getTagName: function() {
          return this._tagName;
        },
        get tagName() { return this.getTagName(); },

        /**
         * Set the tag name for the accessible label for this Node.  DOM element
         * tag names are read-only, so this will require creating a new label
         * element.
         * 
         * @param {string} tagName
         */
        setLabelTagName: function( tagName ) {
          this._labelTagName = tagName;
          this._labelElement = null;

          if ( tagName ) {
            this._labelElement = this.createDOMElement( tagName );
            this._labelElement.id = 'label-' + this._accessibleId;
          }

          this.invalidateAccessibleContent();
        },
        set labelTagName( tagName ) { this.setLabelTagName( tagName ); },

        getLabelTagName: function() {
          return this._labelTagName;
        },
        get labelTagName() { return this.getLabelTagName(); },

        /**
         * Set the tag name for the description. DOM element tag names are
         * read-only, so this will require creating a new descriptiton element.
         * @public
         * 
         * @param {string} tagName
         */
        setDescriptionTagName: function( tagName ) {
          this._descriptionTagName = tagName;
          this._descriptionElement = null;

          if ( tagName ) {
            this._descriptionElement = this.createDOMElement( tagName );
            this._descriptionElement.id = 'description-' + this._accessibleId;
          }

          this.invalidateAccessibleContent();
        },
        set descriptionTagName( tagName ) { this.setDescriptionTagName( tagName ); },

        getDescriptionTagName: function() {
          return this._descriptionTagName;
        },
        get descriptionTagName() { return this.getDescriptionTagName(); },
        
        /**
         * Sets the type for an input element.  Element must have the INPUT tag
         * name. The input attribute is not specified as readonly, so
         * invalidating accessible content is not necessary.
         * 
         * @param {string} inputType
         */
        setInputType: function( inputType ) {
          assert && assert( this._tagName.toUpperCase() === AccessibilityUtil.INPUT_TAG, 'tag name must be INPUT to support inputType' );

          this._inputType = inputType;
          this._domElement.type = inputType;
        },
        set inputType( inputType ) { this.setInputType( inputType ); },

        getInputType: function( inputType ) {
          return this._inputType;
        },
        get inputType() { return this.getInputType(); },

        /**
         * Set whether or not we want to prepend labels above the node's
         * HTML element.  This should only be used if the node has a parent
         * container element. If prepending labels, the label and description
         * elements will be located above the HTML element like:
         *
         * <div id='parent-container'>
         *   <p>Label</p>
         *   <p>Description</p>
         *   <div id="node-content"></div>
         * </div>
         * 
         * @param {boolean} prependLabels
         */
        setPrependLabels: function( prependLabels ) {
          assert && assert( this._parentContainerElement, 'prependLabels requires a parent container element' );
          this._prependLabels = prependLabels;

          this.invalidateAccessibleContent();
        },
        set prependLabels( prependLabels ) { this.setPrependLabels( prependLabels ); },
        
        /**
         * Get whether or not this node adds labels above the accessible content.
         * @return {[type]} [description]
         */
        getPrependLabels: function() {
          return this._prependLabels;
        },
        get prependLabels() { return this.getPrependLabels(); },

        /**
         * Set the parent container tag name.  By specifying this parent container,
         * an element will be created that acts as a container for this node's
         * DOM element and its label and description peers.  For instance, 
         * a button element with a label and description will be contained
         * like the following if the parent container tag name is specified as
         * 'section'.
         *
         * <section id='parent-container-trail-id'>
         *   <button>Press me!</button>
         *   <p>Button label</p>
         *   <p>Button description</p>
         * </section>
         * 
         * @param {string} tagName - tag name for the parent container
         */
        setParentContainerTagName: function( tagName ) {
          this._parentContainerTagName = tagName;
          this._parentContainerElement = this.createDOMElement( tagName );
          this._parentContainerElement.id = 'parent-container-' + this._accessibleId;

          this.invalidateAccessibleContent();
        },
        set parentContainerTagName( tagName ) { this.setParentContainerTagName( tagName ); },

        /**
         * Get the tag name for the parent container element.
         * 
         * @param  {string} tagName
         * @return {string}         
         */
        getParentContainerTagName: function( tagName ) {
          return this._parentContainerTagName;
        },
        get parentContainerTagName() { return this.getParentContainerTagName(); },

        /**
         * Get the parent container element, returning null if none exists.
         * @public (scenery-internal)
         * 
         * @return {HTMLElement|null}
         */
        getParentContainerElement: function() {
          return this._parentContainerElement;
        },
        get parentContainerElement() { return this.getParentContainerElement(); },

        /**
         * Set the label for the this node.  The label can be added in one of
         * four ways:
         *   - As an inline text with the 'aria-label' attribute.
         *   - As a 'label' ellement with the 'for' attribute pointing to the
         *     node's DOM element.
         *   - As inner text on the Node's DOM element itself.
         *   - As a separate DOM element positioned as a peer or child of this
         *     node's DOM element.
         *     
         * The way in which the label is added to the Node is dependent on the 
         * _labelTagName, _useAriaLabel, and whether the node's DOM element
         * supports inner text.
         *
         * @param {string} label
         */
        setAccessibleLabel: function( label ) {
          this._accessibleLabel = label;

          if ( this._useAriaLabel ) {
            this.setAccessibleAttribute( 'aria-label', this._accessibleLabel );
          }
          else if ( AccessibilityUtil.elementSupportsInnerText( this._domElement ) ) {
            this._domElement.innerText = this._accessibleLabel;
          }
          else if ( this._labelTagName ) {
            assert && assert( this._labelElement, 'label element must have been created' );

            // the remaining methods require a new DOM element
            this._labelElement.textContent = this._accessibleLabel;

            // if using a label element it must point to the dom element
            if ( this._labelTagName && this._labelTagName.toUpperCase() === AccessibilityUtil.LABEL_TAG ) {
              this._labelElement.setAttribute( 'for', this._accessibleId );
            }
          }

        },
        set accessibleLabel( label ) { this.setAccessibleLabel( label ); },

        getAccessibleLabel: function() {
          return this._accessibleLabel;
        },
        get accessibleLabel() { return this.getAccessibleLabel(); },

        /**
         * Set the description content for this node's DOM element. A description
         * element must exist and that element must support inner text.  If a
         * description element does not exist yet, we assume that a default paragraph
         * should be used.
         * 
         * @param {string} textContent
         */
        setAccessibleDescription: function( textContent ) {
          this._accessibleDescription = textContent;

          // if there is no description element, assume that a paragraph element should be used
          if ( !this.descriptionElement ) {
            this.setDescriptionTagName( 'p' );
          }

          assert && assert( AccessibilityUtil.elementSupportsInnerText( this._descriptionElement ), 'description element must support inner text' );
          this._descriptionElement.textContent = this._accessibleDescription;

        },
        set accessibleDescription( textContent ) { this.setAccessibleDescription( textContent ); },

        getAccessibleDescription: function() {
          return this._accessibleDescription;
        },
        get accessibleDescription() { return this.getAccessibleDescription(); },

        /**
         * Set the ARIA role for this node's DOM element. According to the W3C,
         * the ARIA role is read-only for a DOM element.  So this will create a
         * new DOM element for this Node with the desired role, and replace the
         * old element in the DOM.
         * @public
         *
         * @param {string} ariaRole - role for the element, see https://www.w3.org/TR/html-aria/#allowed-aria-roles-states-and-properties
         *                            for a list of roles, states, and properties.
         */
        setAriaRole: function( ariaRole ) {
          this._ariaRole = ariaRole;
          this.setAccessibleAttribute( 'role', ariaRole );

          this.invalidateAccessibleContent();
        },
        set ariaRole( ariaRole ) { this.setAriaRole( ariaRole ); },

        /**
         * Get the ARIA role representing this node.
         * @public
         * @return {string}
         */
        getAriaRole: function() {
          return this._ariaRole;
        },
        get ariaRole() { return this.getAriaRole(); },

        /**
         * Sets whether or not to use the 'aria-label' attribute for labelling 
         * the node's DOM element. By using the 'aria-label' attribute, the
         * label will be read on focus, but will not be available with the
         * virtual cursor.  If there is label content, we reset the accessible
         * label.
         * @public
         * 
         * @param {string} useAriaLabel
         */
        setUseAriaLabel: function( useAriaLabel ) {
          this._useAriaLabel = useAriaLabel;
          if ( this._labelElement ) {
            // if we previously had a label element, remove it
            
            this._labelElement.parentNode && this._labelElement.parentNode.removeChild( this._labelElement );
            this._labelElement = null;
            this._labelTagName = null;
          }

          if ( this._accessibleLabel ) {
            this.setAccessibleLabel( this._accessibleLabel );
          }

          this.invalidateAccessibleContent();
        },
        set useAriaLabel( useAriaLabel ) { this.setUseAriaLabel( useAriaLabel ); },

        getUseAriaLabel: function() {
          return this._useAriaLabel;
        },
        get useAriaLabel() { return this.getUseAriaLabel(); },

        /**
         * Set the focus highlight for this node.
         * @public
         * 
         * @param {Node|Shape|string.<'invisible'>} focusHighlight
         */
        setFocusHighlight: function( focusHighlight ) {
          this._focusHighlight = focusHighlight;
          this.invalidateAccessibleContent();
        },
        set focusHighlight( focusHighlight ) { this.setFocusHighlight( focusHighlight ); },

        getFocusHighlight: function() {
          return this._focusHighlight;
        },
        get focusHighlight() { return this.getFocusHighlight(); },

        /**
         * Get an id referencing the description element of this node.  Useful when you want to
         * set aria-describedby on a DOM element that is far from this one in the scene graph or
         * DOM.
         *
         * @return {string}
         * @public
         */
        getDescriptionElementId: function() {
          assert && assert( this._descriptionElement, 'description element must exist in the parallel DOM' );
          return this._descriptionElement.id;
        },
        get descriptionElementId() { return this.getDescriptionElementId(); },

        /**
         * Get an id referencing the label element of this node.  Useful when you want to
         * set aria-labelledby on a DOM element that is far from this one in the scene graph.
         * @public
         *
         * @return {string}
         */
        getLabelElementID: function() {
          assert && assert( this._labelElement, 'description element must exist in the parallel DOM' );
          return this._labelElement.id;
        },

        /**
         * Add the 'aria-describedby' attribute to this node's dom element.
         * @public
         *
         * @param {string} descriptionId - id referencing the description element
         */
        setAriaDescribedById: function( descriptionId ) {
          this._ariaDescribedById = descriptionId;
          this.setAccessibleAttribute( 'aria-describedby', descriptionId );
        },
        set ariaDescribedById( descriptionId ) { this.setAriaDescribedById( descriptionId ); },

        getAriaDescribedById: function() {
          return this._ariaDescribedById;
        },
        get ariaDescribedById() { return this.getAriaDescribedById(); },

        /**
         * Add the 'aria-labelledby' attribute to this node's dom element.
         * @public
         *
         * @param {string} labelId - id referencing the description element
         */
        setAriaLabelledById: function( labelId ) {
          this._ariaLabelledById = labelId;

          // Need to invalidate?
          this.setAccessibleAttribute( 'aria-labelledby', labelId );
        },
        set ariaLabelledById( labelId ) { this.setAriaLabelledById( labelId ); },

        /**
         * Get the id of the element that labels this node's DOM element through
         * the aria-labelledby attribute.
         * @public
         * 
         * @return {string}
         */
        getAriaLabelledById: function() {
          return this._ariaLabelledById;
        },
        get ariaLabelledById() { return this.getAriaLabelledById(); },

        /**
         * If the node is using a list for its description, add a list item to 
         * the end of the list with the text content.  Returns an id so that
         * the element can be referenced if need be.
         * @public
         * 
         * @param  {string} textContent
         * @return {string} - the id of the list item returned for reference
         */
        addDescriptionItem: function( textContent ) {
          assert && assert( this._descriptionElement.tagName === AccessibilityUtil.UNORDERED_LIST_TAG, 'description element must be a list to use addDescriptionItem' );

          var listItem = document.createElement( 'li' );
          listItem.textContent = textContent;
          listItem.id = 'list-item-' + ITEM_NUMBER++;
          this._descriptionElement.appendChild( listItem );

          return listItem.id;
        },

        /**
         * Update the text content of the description item.  The item may not yet be in the DOM, so
         * document.getElementById cannot be used, and the element needs to be found
         * under the description element.
         *
         * @param  {string} itemID - id of the lits item to update
         * @param  {string} description - new textContent for the string
         * @public
         */
        updateDescriptionItem: function( itemID, description ) {
          var listItem = this.getChildElementWithId( this._descriptionElement, itemID );
          assert && assert( this._descriptionElement.tagName === AccessibilityUtil.UNORDERED_LIST_TAG, 'description must be a list to hide list items' );
          assert && assert( listItem, 'No list item in description with id ' + itemID );

          listItem.textContent = description;
        },

        /**
         * Hide or show the desired list item from the screen reader
         *
         * @param {string} itemID - id of the list item to hide
         * @param {boolean} hidden - whether the list item should be hidden
         * @public
         */
        setDescriptionItemHidden: function( itemID, hidden ) {
          var listItem = document.getElementById( itemID );
          assert && assert( this._descriptionElement.tagName === AccessibilityUtil.UNORDERED_LIST_TAG, 'description must be a list to hide list items' );
          assert && assert( listItem, 'No list item in description with id ' + itemID );

          listItem.hidden = hidden;
        },

        /**
         * Hide completely from a screen reader by setting the aria-hidden attribute.
         * If this domElement and its peers have a parent container, it should
         * be hidden so that all peers are hidden as well.
         *
         * @param {boolean} hidden
         * @public
         */
        setAccessibleHidden: function( hidden ) {
          assert && assert( this._domElement, 'node requires an accessible DOM element for setAccessibleHidden' );

          this._accessibleHidden = hidden;
          if ( this._parentContainerElement ) {
            this._parentContainerElement.hidden = hidden;
          }
          else {
            this._domElement.hidden = hidden;
          }
        },
        set accessibleHidden( hidden ) { this.setAccessibleHidden( hidden ); },

        getAccessibleHidden: function() {
          return this._accessibleHidden;
        },
        get accessibleHidden() { return this.getAccessibleHidden(); },

        /**
         * Set the value of an input element.
         * 
         * @param {string} value
         */
        setInputValue: function( value ) {
          assert && assert( _.contains( AccessibilityUtil.FORM_ELEMENTS, this._domElement.tagName ), 'dom element must be a form element to support value' );
          this._domElement.value = value;
        },
        set inputValue( value ) { this.setINputValue( value ); },

        /**
         * Get the value of the element.
         * 
         * @return {string}
         */
        getInputValue: function() {
          assert && assert( _.contains( AccessibilityUtil.FORM_ELEMENTS, this._domElement.tagName ), 'dom element must be a form element to support value' );
          return this._domElement.value;
        },
        get inputValue() { return this.getInputValue(); },
        
        /**
         * Get an array containing all accessible attributes that have been
         * added to this node's DOM element.
         * @public
         * 
         * @return {string[]}
         */
        getAccessibleAttributes: function() {
          return this._accessibleAttributes.slice( 0 ); // defensive copy
        },
        get accessibleAttributes() { return this.getAccessibleAttributes(); },

        /**
         * Set a particular attribute for this node's DOM element, generally to
         * provide extra semantic information for a screen reader.
         *
         * @param  {string} attribute - string naming the attribute
         * @param  {string|boolean} value - the value for the attribute
         * @public
         */
        setAccessibleAttribute: function( attribute, value ) {
          this._accessibleAttributes.push( { attribute: attribute, value: value } );
          this._domElement.setAttribute( attribute, value );
        },

        /**
         * Remove a particular attribute, removing the associated semantic
         * information from the DOM element.
         *
         * @param {string} attribute - name of the attribute to remove
         * @public
         */
        removeAccessibleAttribute: function( attribute ) {
          assert && assert( AccessibilityUtil.hasAttribute( this.domElement, attribute ) );
          this._domElement.removeAttribute( attribute );
        },

        /**
         * Remove all accessible attributes from this node's dom element.
         * @public
         */
        removeAccessibleAttributes: function() {

          // all attributes currently on this node's DOM element
          var attributes = this.getAccessibleAttributes();

          for ( var i = 0; i < attributes.length; i++ ) {
            var attribute = attributes[ i ].attribute;
            this.removeAttribute( attribute );
          }
        },

        /**
         * Make the DOM element explicitly focusable with tab index. Note that
         * native HTML form elements will generally be in the navigation order
         * without explicitly setting focusable.  If these need to be removed
         * from the navigation order, setFocusable( false ).
         * @public
         * 
         * @param {boolean} isFocusable
         */
        setFocusable: function( isFocusable ) {
          this._focusable = isFocusable;
          this._domElement.tabIndex = isFocusable ? 0 : -1;
        },
        set focusable( isFocusable ) { this.setFocusable( isFocusable ); },

        /**
         * Get if this node is focusable by a keyboard.
         *
         * @return {boolean}
         * @public
         */
        getFocusable: function() {
          return this._focusable;
        },
        get focusable() { return this.getFocusable(); },

        /**
         * Focus this node's dom element.
         *
         * @public
         */
        focus: function() {
          assert && assert( this._focusable, 'trying to set focus on a node that is not focusable' );
          assert && assert( !this._accessibleHidden, 'trying to set focus on a node with hidden accessible content' );

          // make sure that the elememnt is in the navigation order
          this.setFocusable( true );
          this._domElement.focus();
        },

        /**
         * Get all 'element' nodes off the parent element, placing them in an
         * array for easy traversal.  Note that this includes all elements, even
         * those that are 'hidden' or purely for structure.
         *
         * @param  {HTMLElement} domElement - parent whose children will be linearized
         * @return {HTMLElement[]}
         * @private
         */
        getLinearDOMElements: function( domElement ) {

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
        },

        /**
         * Get the next focusable element from this node's DOM element.
         * @public
         *
         * @return {HTMLElement}
         */
        getNextFocusable: function() {
          return this.getNextPreviousFocusable( AccessibilityUtil.NEXT );
        },

        /**
         * Get the previous focusable element from this node's DOM element.
         * @public
         *
         * @return {HTMLElement}
         */
        getPreviousFocusable: function() {
          return this.getNextPreviousFocusable( AccessibilityUtil.PREVIOUS );
        },

        /**
         * Get the next or previous focusable element in the parallel DOM,
         * relative to this Node's domElement depending on the parameter.
         * Useful if you need to set focus dynamically or need to prevent
         * default behavior when focus changes.
         * This function should not be used directly, use getNextFocusable() or
         * getPreviousFocusable() instead.
         * @private
         *
         * @param {string} direction - direction of traversal, one of 'NEXT' | 'PREVIOUS'
         * @return {HTMLElement}
         */
        getNextPreviousFocusable: function( direction ) {

          // linearize the DOM or easy traversal
          var linearDOM = this.getLinearDOMElements( document.getElementsByClassName( 'accessibility' )[ 0 ] );

          var activeElement = this._domElement;
          var activeIndex = linearDOM.indexOf( activeElement );
          var delta = direction === AccessibilityUtil.NEXT ? +1 : -1;

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
        },

        /**
         * Get a child element with an id.  This should only be used if the
         * element has not been added to the document yet.  If the element is
         * in the document, document.getElementById is a faster and more
         * conventional option.
         *
         * @param  {HTMLElement} parentElement
         * @param  {string} childId
         * @return {HTMLElement}
         */
        getChildElementWithId: function( parentElement, childId ) {
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
        }
      } );

      /**
       * Invalidate our current accessible content, triggering recomputation
       * of anything that depended on the old accessible content. This can be
       * combined with a client implementation of invalidateAccessibleContent.
       *
       * @protected
       */
      function invalidateAccessibleContent() {

        // clear the parent container if it exists since we will be reinserting labels
        // and the dom element in createPeer
        while ( this._parentContainerElement && this._parentContainerElement.hasChildNodes() ) {
          this._parentContainerElement.removeChild( this._parentContainerElement.lastChild );
        }

        var self = this;
        this.accessibleContent = {
          focusHighlight: this._focusHighlight,
          createPeer: function( accessibleInstance ) {

            var accessiblePeer = new scenery.AccessiblePeer( accessibleInstance, self._domElement, {
              parentContainerElement: self._parentContainerElement
            } );

            // if we have a parent container, add the label element as a child of the container
            // otherwise, add as child of the node's DOM element
            if ( self._labelElement ) {
              if ( self._parentContainerElement ) {
                if ( self._prependLabels && self._parentContainerElement === self._domElement.parentNode ) {
                  self._parentContainerElement.insertBefore( self._labelElement, self._domElement );
                }
                else {
                  self._parentContainerElement.appendChild( self._labelElement );
                }
              }
              else {
                self._domElement.appendChild( self._labelElement );
              }
            }

            // if we have a parent container, add the description element as a child of the container
            // otherwise, add as a child of the node's dom element
            if ( self._descriptionElement ) {
              if ( self._parentContainerElement ) {
                if ( self._prependLabels && self._parentContainerElement === self._domElement.parentNode ) {
                  self._parentContainerElement.insertBefore( self._descriptionElement, self._domElement );
                }
                else {
                  self._parentContainerElement.appendChild( self._descriptionElement );
                }
              }
              else {
                self._domElement.appendChild( self._descriptionElement );
              }
            }

            return accessiblePeer;
          }
        };
      }

      // Patch in a sub-type call if it already exists on the prototype
      if ( proto.invalidateAccessibleContent ) {
        var subtypeInvalidateAccesssibleContent = proto.invalidateAccessibleContent;
        proto.invalidateAccessibleContent = function() {
          subtypeInvalidateAccesssibleContent.call( this );
          invalidateAccessibleContent.call( this );
        };
      }
      else {
        proto.invalidateAccessibleContent = invalidateAccessibleContent;
      }
    }
  };

  scenery.register( 'Accessibility', Accessibility );

  return Accessibility;
} );