// Copyright 2017, University of Colorado Boulder

/**
 * A mixin for Node that implements accessibility by adding HTML content to the parallel DOM.
 *
 * The parallel DOM is an HTML structure that provides semantics for assistive technologies. For web content to be
 * accessible, assistive technologies require HTML markup, which is something that pure graphical content does not
 * include.  This mixin implements the accessible HTML content for any node in the scene graph.
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
 * a peer description paragraph.  This mixin supports this structure with the 'parentContainerElement' option.  In
 * this example, the parentContainerElement is the div, while the description is added as a child under the button
 * node's domElement.
 *
 * For additional accessibility options, please see the options listed in ACCESSIBILITY_OPTION_KEYS. For more
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
  var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );

  // constants
  var globalListItemCounter = 0;

  // type flag, ELEMENT_NODE isn't defined on HTMLElement in Safari
  var ELEMENT_NODE = HTMLElement.ELEMENT_NODE || HTMLElement.prototype.ELEMENT_NODE;

  // specific HTML tag names
  var INPUT_TAG = 'INPUT';
  var LABEL_TAG = 'LABEL';
  var UNORDERED_LIST_TAG = 'UL';
  var BUTTON_TAG = 'BUTTON';
  var TEXTAREA_TAG = 'TEXTAREA';
  var SELECT_TAG = 'SELECT';
  var OPTGROUP_TAG = 'OPTGROUP';
  var DATALIST_TAG = 'DATALIST';
  var OUTPUT_TAG = 'OUTPUT';
  var A_TAG = 'A';

  // these elements are typically associated with forms, and support certain attributes
  var FORM_ELEMENTS = [ INPUT_TAG, BUTTON_TAG, TEXTAREA_TAG, SELECT_TAG, OPTGROUP_TAG, DATALIST_TAG, OUTPUT_TAG, A_TAG ];

  // these elements do not have a closing tag, so they won't support features like innerHTML
  var ELEMENTS_WITHOUT_CLOSING_TAG = [ INPUT_TAG ];

  // these elements require a minimum width to be visible in Safari
  var ELEMENTS_REQUIRE_WIDTH = [ INPUT_TAG, A_TAG ];

  // these custom attributes flag that the DOM element needs special handling during invalidation
  var DATA_FLAG = 'data-';
  var RELATION_ATTRIBUTES = [ 'aria-labelledby', 'aria-describedby' ];

  // these events change the input value on the dom element
  var INPUT_CHANGE_EVENTS = [ 'input', 'change' ];

  var ACCESSIBILITY_OPTION_KEYS = [
    'tagName', // Sets the tag name for the DOM element representing this node in the parallel DOM
    'inputType', // Sets the input type for the representative DOM element, only relevant if tagname is 'input'
    'inputValue', // Sets the input value for the representative DOM element, only relevant if tagname is 'input'
    'parentContainerTagName', // Sets the tag name for an element that contains this node's DOM element and its peers
    'labelTagName', // Sets the tag name for the DOM element labelling this node, usually a paragraph
    'descriptionTagName', // Sets the tag name for the DOM element describing this node, usually a paragraph
    'focusHighlight', // Sets the focus highlight for the node, see setFocusHighlight()
    'focusHighlightLayerable', // Flag to determine if the focus highlight node can be layered in the scene graph, see setFocusHighlightLayerable()
    'accessibleLabel', // Set the label content for the node, see setAccessibleLabel()
    'accessibleDescription', // Set the description content for the node, see setAccessibleDescription()
    'accessibleHidden', // Sets wheter or not the node's DOM element is hidden in the parallel DOM
    'accessibleContentDisplayed', // sets whether or not the accessible content of the node (and its subtree) is displayed, see setAccessibleContentDisplayed()
    'focusable', // Sets whether or not the node can receive keyboard focus
    'useAriaLabel', // Sets whether or not the label will use the 'aria-label' attribute, see setUseAriaLabel()
    'ariaRole', // Sets the ARIA role for the DOM element, see setAriaRole() for documentation
    'parentContainerAriaRole', // Sets the ARIA role for the parent container DOM element, see setParentContainerAriaRole()
    'ariaDescribedByElement', // Sets a description relationship for this node's DOM element by id, see setAriaDescribedByElement()
    'ariaLabelledByElement', // Sets a label relationship with another element in the DOM by id, see setAriaLabelledByElement()
    'prependLabels'// Sets whether we want to prepend labels above the node's HTML element, see setPrependLabels()
  ];

  var Accessibility = {

    /**
     * Given the constructor for Node, mix accessibility functions into the prototype
     *
     * @param {function} type - the constructor for Node
     */
    mixin: function( type ) {
      var proto = type.prototype;

      /**
       * These properties and methods are put directly on the prototype of nodes that have Accessibility mixed in.
       */
      extend( proto, {

        /**
         * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in
         * the order they will be evaluated.  Beware that order matters for accessibility options, changing the order
         * of ACCESSIBILITY_OPTION_KEYS could break the mixin.
         * @protected
         *
         * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
         *       cases that may apply.
         */
        _mutatorKeys: ACCESSIBILITY_OPTION_KEYS.concat( proto._mutatorKeys ),

        /**
         * This should be called in the constructor to initialize the accessibility-specific parts of Node.
         * @protected
         */
        initializeAccessibility: function() {

          //REVIEW: Including the 'null' case in type documentation here would help.

          // @private {string} - the HTML tag name of the element representing this node in the DOM
          this._tagName = null;

          // @private {string} - the HTML tag name for a parent container element for this node in the DOM. This
          // parent container will contain the node's DOM element, as well as peer elements for any label or description
          // content. See setParentContainerTagName() for more documentation.
          this._parentContainerTagName = null;

          // @private {string} - the HTML tag name for the label element that will contain the label content for
          // this dom element. There are ways in which you can have a label without specifying a label tag name,
          // see setAccessibleLabel() for the list of ways.
          this._labelTagName = null;

          // @private {string} - the HTML tag name for the description element that will contain descsription content
          // for this dom element. If a description is set before a tag name is defined, a paragraph element
          // will be created for the description.
          this._descriptionTagName = null;

          // @private {string} - the type for an element with tag name of INPUT.  This should only be used
          // if the element has a tag name INPUT.
          this._inputType = null;

          // @private {string} - the value of the input, only relevant if the tag name is of type "INPUT".
          this._inputValue = null;

          // @private {HTMLElement} - the HTML element representing this node in the DOM.
          this._domElement = null;

          // @private {HTMLElement} - the HTML element that contains label content for this node in the DOM. There
          // are ways to create labels without the label element, see setAccessibleLabel() for more information.
          this._labelElement = null;

          // @private {HTMLElement} - the HTML element that contains the description content for this node in the DOM.
          this._descriptionElement = null;

          // @private {HTMLElement} - the HTML element that will contain this node's DOM element, and possibly its peer
          // description and label elements.  This element will let you structure the label/description content above
          // or below this node's DOM element.  see setParentContainerTagName() and setPrependLabels()
          this._parentContainerElement = null;

          // @private {boolean} - determines whether or not labels should be prepended above the node's DOM element. This
          // should only be used if the element has a parentContainerElement, as the labels are sorted relative to the
          // node's DOM element under the parent container.
          this._prependLabels = null;

          // @private {array.<Object> - array of attributes that are on the node's DOM element.  Objects will have the
          // form { attribute:{string}, value:{string|number} }
          this._accessibleAttributes = [];

          // @private {string} - the label content for this node's DOM element.  There are multiple ways that a label
          // can be associated with a node's dom element, see setAccessibleLabel() for more documentation
          this._accessibleLabel = null;

          // @private {string} - the description content for this node's DOM element.
          this._accessibleDescription = null;

          // @private {boolean} - if true, the aria label will be added as an inline attribute on the node's DOM
          // element.  This will determine how the label content is associated with the DOM element, see
          // setAccessibleLabel() for more information
          this._useAriaLabel = null;
          //REVIEW: This is marked as boolean, but initialized to null?

          // @private {string} - the ARIA role for this node's DOM element, added as an HTML attribute.  For a complete
          // list of ARIA roles, see https://www.w3.org/TR/wai-aria/roles.  Beware that many roles are not supported
          // by browsers or assistive technologies, so use vanilla HTML for accessibility semantics where possible.
          this._ariaRole = null;

          // @private {string} - the ARIA role for the parent container element, added as an HTML attribute. For a
          // complete list of ARIA roles, see https://www.w3.org/TR/wai-aria/roles. Beware that many roles are not
          // supported by browsers or assistive technologies, so use vanilla HTML for accessibility semantics where
          // possible.
          this._parentContainerAriaRole = null;

          // @private {string} - the HTML element that will act as the description for this node's
          // DOM element. The id is added to this node's DOM element with the 'aria-describedby' attribute.
          // The description element can be anywhere in the document.  The behavior for aria-describedby is such that
          // content under the description element will be read whenever the element with the aria-describedby attribute
          // receives focus.
          this._ariaDescribedByElement = null;

          // @private {string} - the HTML element that will act as the label for this node's
          // DOM element. The id is added to this node's DOM element as the 'aria-labelledby' attribute.
          // The label element can be anywhere in the document.  The behavior for aria-labelledby is such
          // that the content under the label element will be read whenever the element with the aria-labelledby
          // attribute receives focus.  There are multiple ways to add a label to a node, see setAccessibleLabel().
          this._ariaLabelledByElement = null;

          // @private {boolean} - whether or not this node's DOM element can receive focus from tab navigation.
          // Sets the tabIndex attribute on the node's DOM element.  Setting to false will not remove the node's DOM
          // from the document, but will ensure that it cannot receive focus by pressing 'tab'.  Several HTMLElements
          // (such as HTML form elements) can be focusable by default, without setting this property.
          this._focusable = null;

          // @private {Shape|Node|string.<'invisible'>} - the focus highlight that will surround this node when it
          // is focussed.  By default, the focus highlight will be a pink rectangle that surrounds the Node's local
          // bounds.
          this._focusHighlight = null;

          // @private {boolean} - Whether or not the accessible content will be hidden from the browser and assistive
          // technologies.  When accessibleHidden is true, the node's DOM element will not be focusable, and it cannot
          // be found by the assistive technology virtual cursor. For more information on how assistive technologies
          // read with the virtual cursor see
          // http://www.ssbbartgroup.com/blog/how-windows-screen-readers-work-on-the-web/
          this._accessibleHidden = null;

          // @private {boolean} - Whether or not the accessible content will be hidden from the browser and assistive
          // technologies.  When accessible content is not displayed, the node will not be focusable, and it cannot
          // be found by assistive technology with the virtual cursor.  Content should almost always be hidden with 
          // setAccessibleHidden(), see that function and setAccessibleContentDisplayed() for more information.
          this._accessibleContentDisplayed = true;

          // @private {string} - An id used for the purposes of accessibility.  The accessible id is a string, since
          // the DOM API frequently uses string identifiers to reference other elements in the document.
          // The accessible id will be null until all instancess have been added to the instance tree. Each instance
          // must have a unique id or else the document would contain duplicate ids.
          this._accessibleId = null;

          // @private {string} - An id used to reference the label element.  This will only be defined if a label
          // element exists, see setLabelTagName() for the ways a label can be added.  This is useful for setting
          // attributes on the DOM element relative to the label element such as "aria-labelledby".
          this._labelElementId = null;

          // @private {string} - An id used to reference the description element.  This will only be defined if a
          // description exists, set setDescriptionTagName().  This is useful for setting attributes on this node's
          // DOM element relative to the description element, such as "aria-describedby".
          this._descriptionElementId = null;

          // @private {Array.<Function>} - For accessibility input handling {keyboard/click/HTML form}
          this._accessibleInputListeners = [];

          // @private {Array.<Function>} - Accessibility input listeners that need to be added to the DOM element, see
          // addDOMEventListeners(). Once added, the listener will be moved to this._accessibleInputListeners.
          this._accessibleInputListenersToAdd = [];

        },

        /**
         * Add DOM event listeners contained in the accessibleInput directly to the DOM element. Never to be used
         * directly, see addAccessibilityInputListener().
         * @private
         *
         * REVIEW: In general, functions like this shouldn't mutate the object provided. Currently,
         * calling it on two consecutive nodes will double-wrap the listener. A typical pattern would be to return an
         * object with the actually-added listeners, so that they could be removed with removeDOMEventListeners.
         *
         * @param {Object} accessibleInput
         */
        addDOMEventListeners: function( accessibleInput ) {
          var self = this;

          var listenerIndex = self._accessibleInputListenersToAdd.indexOf( accessibleInput );
          assert && assert( listenerIndex >= 0, 'listener must have been in self._accessibleInputListenersToAdd' );

          for ( var event in accessibleInput ) {
            if ( accessibleInput.hasOwnProperty( event ) ) {
              self._domElement.addEventListener( event, accessibleInput[ event ] );
            }
          }

          self._accessibleInputListenersToAdd.splice( listenerIndex, 1 );
          self._accessibleInputListeners.push( accessibleInput );
        },

        /**
         * Remove a DOM event listener contained in an accesssibleInput.  Never to be used directly, see
         * removeAccessibilityInputListener().
         * @private
         *
         * @param {Object} accessibleInput
         */
        removeDOMEventListeners: function( accessibleInput ) {
          var self = this;

          assert && assert( this._domElement, 'dom element must be defined to remove event listeners' );
          for ( var event in accessibleInput ) {
            if ( accessibleInput.hasOwnProperty( event ) ) {
              self._domElement.removeEventListener( event, accessibleInput[ event ] );
            }
          }
        },

        /**
         * Adds an accessible input listener.
         * @public
         *
         * The listener's keys should be DOM event names, and the values should be functions to be called when that
         * event is fired on the dom element.
         *
         * @param {Object} listener
         * @returns {Object} - the actually added listener, so it can be removed via removeAccessibleInputListener
         */
        addAccessibleInputListener: function( accessibleInput ) {
          var self = this;

          // event changes the input value, wrap the listener with a function that will handle this
          var addedAccessibleInput = {};
          for ( var event in accessibleInput ) {
            if ( accessibleInput.hasOwnProperty( event ) ) {
              addedAccessibleInput[ event ] = accessibleInput[ event ];
              if ( _.includes( INPUT_CHANGE_EVENTS, event ) ) {
                addedAccessibleInput[ event ] = function( e ) {
                  self._inputValue = e.target.value;
                  accessibleInput[ e.type ]();
                };
              }
            }
          }

          // don't allow listeners to be added multiple times
          // REVIEW: Presumably passing an already-existing listener would be an assertion failure?
          var listenerAlreadyAdded = ( _.indexOf( this._accessibleInputListeners, addedAccessibleInput ) > 0 );
          var listenerWillBeAdded = ( _.indexOf( this._accessibleInputListenersToAdd, addedAccessibleInput ) > 0 );
          if ( !listenerAlreadyAdded && !listenerWillBeAdded ) {
            this._accessibleInputListenersToAdd.push( addedAccessibleInput );
          }

          this.invalidateAccessibleContent();
          return addedAccessibleInput;
        },

        /**
         * Removes an input listener that was previously added with addAccessibleInputListener.
         * @public
         *
         * @param {Object} accessibleInput
         * @returns {Node} - Returns 'this' reference, for chaining
         */
        removeAccessibleInputListener: function( accessibleInput ) {

          // ensure the listener is in our list, or will be added in invalidation
          var addedIndex = _.indexOf( this._accessibleInputListeners, accessibleInput );
          var toBeAddedIndex = _.indexOf( this._accessibleInputListenersToAdd, accessibleInput );
          var listenerAdded = addedIndex > -1;
          var listenerToBeAdded = toBeAddedIndex > -1;
          assert && assert( listenerAdded || listenerToBeAdded, 'accessibleInput was not added' );

          if ( listenerAdded ) {
            this._accessibleInputListeners.splice( addedIndex, 1 );

            // remove the event listeners from the DOM element
            this.removeDOMEventListeners( accessibleInput );
            this.invalidateAccessibleContent();
          }
          else {

            // listener hasn't been added yet, remove from list of listeners to add
            this._accessibleInputListenersToAdd.splice( toBeAddedIndex, 1 );
          }

          return this;
        },

        /**
         * Returns a copy of all input listeners related to accessibility.
         * @public
         *
         * @returns {Array.<Object>}
         */
        getAccessibleInputListeners: function() {
          return this._accessibleInputListeners.slice( 0 ); // defensive copy
        },
        get accessibleInputListeners() { return this.getAccessibleInputListeners(); },

        /**
         * Get the accesible id for this node. It is a string since elemetns are are generally referenced by string
         * with the the DOM API.
         *
         * REVIEW: How can this guarantee it is a string, when it is initialized as null?
         *         new Node().accessibleId returns null.
         *
         * @returns {string}
         */
        getAccessibleId: function() {
          return this._accessibleId;
        },
        get accessibleId() { return this.getAccessibleId(); },

        /**
         * Get HTML element representing this node in the document.
         * @public
         *
         * REVIEW: How can this guarantee it is an HTMLElement, when it is initialized as null?
         *         new Node().domElement returns null.
         *
         * @returns {HTMLElement}
         */
        getDomElement: function() {
          return this._domElement;
        },
        get domElement() { return this.getDomElement(); },

        /**
         * Set the tag name representing this element in the DOM. DOM element tag names are read-only, so this
         * function will create a new DOM element for the Node and reset the accessible content.
         *
         * REVIEW: Setting the tag name multiple times results in incorrect behavior with many functions, e.g.:
         *   var node = new scenery.Node();
         *   node.tagName = 'div';
         *   node.focusable = true;
         *   node.domElement.tabIndex // 0 (as expected)
         *   node.tagName = 'p';
         *   node.domElement.tabIndex // -1 (yikes!, even when node.focusable returns true)
         *
         * @param {string} tagName
         */
        setTagName: function( tagName ) {
          assert && assert( tagName === null || typeof tagName === 'string' );

          this._tagName = tagName;
          if ( tagName ) {
            this._domElement = createElement( tagName, this._focusable );

            // Safari requires that certain input elements have width, otherwise it will not be keyboard accessible
            if ( _.includes( ELEMENTS_REQUIRE_WIDTH, tagName.toUpperCase() ) ) {
              this._domElement.style.width = '1px';
            }
          }

          this.invalidateAccessibleContent();
        },
        set tagName( tagName ) { this.setTagName( tagName ); },

        /**
         * Get the tag name of the DOM element representing this node for accessibility.
         * @public
         *
         * REVIEW: Return type should include null, since new scenery.Node().tagName is null.
         *
         * @returns {string}
         */
        getTagName: function() {
          return this._tagName;
        },
        get tagName() { return this.getTagName(); },

        /**
         * Set the tag name for the accessible label for this Node.  DOM element tag names are read-only, so this will
         * require creating a new label element.
         *
         * REVIEW: Same problem with after-the-fact modification as tagName:
         *   var node = new scenery.Node()
         *   node.tagName = 'div';
         *   node.labelTagName = 'p'
         *   node.accessibleLabel = 'Label';
         *   node.getLabelElement() // <p>Label</p>
         *   node.labelTagName = 'div';
         *   node.getLabelElement() // <div></div> -- NO label specified, even though accessibleLabel is still set
         *
         * REVIEW: null used in unit tests, so this should be marked as accepting null
         *
         * @param {string} tagName
         */
        setLabelTagName: function( tagName ) {
          this._labelTagName = tagName;
          this._labelElement = null;

          // REVIEW: Is null supported, or is this meant to check for empty strings?
          if ( tagName ) {
            this._labelElement = createElement( tagName, false );
          }

          this.invalidateAccessibleContent();
        },
        set labelTagName( tagName ) { this.setLabelTagName( tagName ); },

        /**
         * Get the label element HTML tag name.
         * @public
         *
         * REVIEW: Return type should include null, since new scenery.Node().labelTagName is null.
         *
         * @returns {string}
         */
        getLabelTagName: function() {
          return this._labelTagName;
        },
        get labelTagName() { return this.getLabelTagName(); },

        /**
         * Set the tag name for the description. HTML element tag names are read-only, so this will require creating
         * a new HTML element, and inserting it into the DOM.
         * @public
         *
         * REVIEW: Has same issue with setting tagName and labelTagName (see those review comments)
         *
         * @param {string} tagName
         */
        setDescriptionTagName: function( tagName ) {
          this._descriptionTagName = tagName;
          this._descriptionElement = null;

          // REVIEW: Is null supported, or is this meant to check for empty strings?
          if ( tagName ) {
            this._descriptionElement = createElement( tagName, false );
          }

          this.invalidateAccessibleContent();
        },
        set descriptionTagName( tagName ) { this.setDescriptionTagName( tagName ); },

        /**
         * Get the HTML get name for the description element.
         * @public
         *
         * @returns {string|null}
         */
        getDescriptionTagName: function() {
          return this._descriptionTagName;
        },
        get descriptionTagName() { return this.getDescriptionTagName(); },

        /**
         * Sets the type for an input element.  Element must have the INPUT tag name. The input attribute is not
         * specified as readonly, so invalidating accessible content is not necessary.
         *
         * @param {string} inputType
         */
        setInputType: function( inputType ) {
          assert && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tag name must be INPUT to support inputType' );

          this._inputType = inputType;
          this._domElement.type = inputType;
        },
        set inputType( inputType ) { this.setInputType( inputType ); },

        /**
         * Get the input type. Input type is only relevant if this node's DOM element has tag name "INPUT".
         * @public
         *
         * @returns {string}
         */
        getInputType: function() {
          return this._inputType;
        },
        get inputType() { return this.getInputType(); },

        /**
         * Set whether or not we want to prepend labels above the node's HTML element.  If the node does not have
         * a parent container element, one will be created. If prepending labels, the label and description elements
         * will be located above the HTML element like:
         *
         * <div id='parent-container'>
         *   <p>Label</p>
         *   <p>Description</p>
         *   <div id="node-content"></div>
         * </div>
         *
         * By default, label and description elements are placed below the node's HTML element.
         *
         * @param {boolean} prependLabels
         */
        setPrependLabels: function( prependLabels ) {

          // if there isn't a parent container element, create one so labels can be prepended
          if ( !this._parentContainerElement ) {
            this.parentContainerTagName = 'div';
          }

          this._prependLabels = prependLabels;

          this.invalidateAccessibleContent();
        },
        set prependLabels( prependLabels ) { this.setPrependLabels( prependLabels ); },

        /**
         * Get whether or not this node adds labels and descriptions above the representative DOM element.
         * @public
         *
         * @returns {boolean}
         */
        getPrependLabels: function() {
          return this._prependLabels;
        },
        get prependLabels() { return this.getPrependLabels(); },

        /**
         * Set the parent container tag name.  By specifying this parent container, an element will be created that
         * acts as a container for this node's DOM element and its label and description peers.  For instance, a button
         * element with a label and description will be contained like the following if the parent container tag name
         * is specified as 'section'.
         *
         * <section id='parent-container-trail-id'>
         *   <button>Press me!</button>
         *   <p>Button label</p>
         *   <p>Button description</p>
         * </section>
         *
         * @param {string} tagName
         */
        setParentContainerTagName: function( tagName ) {
          this._parentContainerTagName = tagName;
          this._parentContainerElement = createElement( tagName, false /* not focusable */ );

          this.invalidateAccessibleContent();
        },
        set parentContainerTagName( tagName ) { this.setParentContainerTagName( tagName ); },

        /**
         * Get the tag name for the parent container element.
         *
         * @returns {string}
         */
        getParentContainerTagName: function() {
          return this._parentContainerTagName;
        },
        get parentContainerTagName() { return this.getParentContainerTagName(); },

        /**
         * Get the parent container element, returning null if none exists.
         * @public (scenery-internal)
         *
         * @returns {HTMLElement|null}
         */
        getParentContainerElement: function() {
          return this._parentContainerElement;
        },
        get parentContainerElement() { return this.getParentContainerElement(); },

        /**
         * Set the label for the this node.  The label can be added in one of four ways:
         *   - As an inline text with the 'aria-label' attribute.
         *   - As a 'label' ellement with the 'for' attribute pointing to the
         *     node's DOM element.
         *   - As inner text on the Node's DOM element itself.
         *   - As a separate DOM element positioned as a peer or child of this
         *     node's DOM element.
         *
         * The way in which the label is added to the Node is dependent on the label tag name, whether we use the
         * aria-label attribute, and whether the node's DOM element supports inner HTML.
         *
         * @param {string} label
         */
        setAccessibleLabel: function( label ) {
          this._accessibleLabel = label;

          if ( this._useAriaLabel ) {
            this.setAccessibleAttribute( 'aria-label', this._accessibleLabel );
          }
          else if ( this._labelTagName ) {
            assert && assert( this._labelElement, 'label element must have been created' );

            // the remaining methods require a new DOM element
            setTextContent( this._labelElement, this._accessibleLabel );

            // if using a label element it must point to the dom element
            if ( this._labelTagName.toUpperCase() === LABEL_TAG ) {
              this.invalidateAccessibleContent();
            }
          }
          else if ( elementSupportsInnerHTML( this._domElement ) ) {
            setTextContent( this._domElement, this._accessibleLabel );
          }
        },
        set accessibleLabel( label ) { this.setAccessibleLabel( label ); },

        /**
         * Get the label content for this node's DOM element.
         *
         * @returns {string}
         */
        getAccessibleLabel: function() {
          return this._accessibleLabel;
        },
        get accessibleLabel() { return this.getAccessibleLabel(); },

        /**
         * Set the description content for this node's DOM element. A description element must exist and that element
         * must support inner HTML.  If a description element does not exist yet, we assume that a default paragraph
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
          setTextContent( this._descriptionElement, this._accessibleDescription );
        },
        set accessibleDescription( textContent ) { this.setAccessibleDescription( textContent ); },

        /**
         * Get the accessible description content that is describing this Node.
         *
         * @returns {string}
         */
        getAccessibleDescription: function() {
          return this._accessibleDescription;
        },
        get accessibleDescription() { return this.getAccessibleDescription(); },

        /**
         * Set the ARIA role for this node's DOM element. According to the W3C, the ARIA role is read-only for a DOM
         * element.  So this will create a new DOM element for this Node with the desired role, and replace the old
         * element in the DOM.
         * @public
         *
         * @param {string} ariaRole - role for the element, see
         *                            https://www.w3.org/TR/html-aria/#allowed-aria-roles-states-and-properties
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
         *
         * @returns {string}
         */
        getAriaRole: function() {
          return this._ariaRole;
        },
        get ariaRole() { return this.getAriaRole(); },

        /**
         * Set the ARIA role for this node's parent container element.  According to the W3C, the ARIA role is read-only
         * for a DOM element. This will create a new DOM element for the parent container with the desired role, and
         * replace it in the DOM.
         * @public
         *
         * @param {string} ariaRole - role for the element, see
         *                            https://www.w3.org/TR/html-aria/#allowed-aria-roles-states-and-properties
         *                            for a lsit of roles, states, and properties.
         */
        setParentContainerAriaRole: function( ariaRole ) {
          this._parentContainerAriaRole = ariaRole;
          this.invalidateAccessibleContent();
        },
        set parentContainerAriaRole( ariaRole ) { this.setParentContainerAriaRole( ariaRole ); },

        /**
         * Get the ARIA role assigned to the parent container element.
         * @public
         * @returns {string|null}
         */
        getParentContainerAriaRole: function() {
          return this._parentContainerAriaRole;
        },
        get parentcontainerAriaRole() { return this.getParentContainerAriaRole(); },

        /**
         * Sets whether or not to use the 'aria-label' attribute for labelling  the node's DOM element. By using the
         * 'aria-label' attribute, the label will be read on focus, but will can not be found with the
         * virtual cursor.
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

          // if a label is defined, reset the label content
          if ( this._accessibleLabel ) {
            this.setAccessibleLabel( this._accessibleLabel );
          }

          this.invalidateAccessibleContent();
        },
        set useAriaLabel( useAriaLabel ) { this.setUseAriaLabel( useAriaLabel ); },

        /**
         * Get whether or not we are using an aria-label to label this node's HTML element.
         *
         * @returns {boolean}
         */
        getUseAriaLabel: function() {
          return this._useAriaLabel;
        },
        get useAriaLabel() { return this.getUseAriaLabel(); },

        /**
         * Set the focus highlight for this node. By default, the focus highlight will be a pink rectangle that
         * surrounds the node's local bounds.  If focus highlight is set to 'invisible', the node will not have
         * any highlighting when it receives focus.
         * @public
         *
         * @param {Node|Shape|string.<'invisible'>} focusHighlight
         */
        setFocusHighlight: function( focusHighlight ) {
          this._focusHighlight = focusHighlight;

          var isFocussed = false;
          if ( this.isFocussed() ) {
            isFocussed = true;
          }
          this.invalidateAccessibleContent();

          // Reset the focus after invalidating the content.
          isFocussed && this.focus();

        },
        set focusHighlight( focusHighlight ) { this.setFocusHighlight( focusHighlight ); },

        /**
         * Get the focus highlight for this node.
         * @public
         *
         * @returns {Node|Shape|string<'invisible'>}
         */
        getFocusHighlight: function() {
          return this._focusHighlight;
        },
        get focusHighlight() { return this.getFocusHighlight(); },

        /**
         * Setting a flag to break default and allow the focus highlight to be (z) layered into the scene graph.
         * TODO: We may want to eventually handle the case of setting this flag while the node is currently focused.
         * @param {Boolean} focusHighlightLayerable
         */
        setFocusHighlightLayerable: function( focusHighlightLayerable ) {
          this._focusHighlightLayerable = focusHighlightLayerable;
          this.invalidateAccessibleContent();
        },
        set focusHighlightLayerable( focusHighlightLayerable ) { this.setFocusHighlightLayerable( focusHighlightLayerable ); },

        /**
         * Get the flag for if this node is layerable in the scene graph (or if it is always on top, like the default).
         * @public
         *
         * @returns {Boolean}
         */
        getFocusHighlightLayerable: function() {
          return this._focusHighlightLayerable;
        },
        get focusHighlightLayerable() { return this.getFocusHighlightLayerable(); },

        /**
         * Get the description element that holds the description content for this node.
         * @public
         *
         * @returns {HTMLElement|null}
         */
        getDescriptionElement: function() {
          return this._descriptionElement;
        },
        get descriptionElement() { return this.getDescriptionElement(); },

        /**
         * Get the label element that holds the label content for this node.
         * @public
         * @returns {HTMLElement|null}
         */
        getLabelElement: function() {
          return this._labelElement;
        },
        //REVIEW: Why no ES5 getter?

        /**
         * Set the 'aria-describedby' element for this node's DOM element. The value of the 'aria-describedby'
         * attribute is a string id that references the desired HTML element.  Upon focus, a screen reader should
         * read the content under the HTML element referenced by the 'aria-describedby' id. Exact behavior will depend
         * slightly on user agent.
         * @public
         *
         * @param {HTMLElement} descriptionElement
         */
        setAriaDescribedByElement: function( descriptionElement ) {
          assert && assert( this._domElement, 'HTML element required for aria-describedby attribute, see setTagName' );
          assert && assert( descriptionElement.nodeType === ELEMENT_NODE, 'HTML element required' );

          this._ariaDescribedByElement = descriptionElement;
          this.invalidateAccessibleContent();
        },
        set ariaDescribedByElement( descriptionElement ) { this.setAriaDescribedByElement( descriptionElement ); },

        /**
         * Get the id that is the value of the 'aria-describedby' attribute.  See setAriaDescribedBy() for details
         * about the 'aria-describedby' attribute.
         * @returns {[type]} [description]
         */
        getAriaDescribedByElement: function() {
          return this._ariaDescribedByElement;
        },
        get ariaDescribedByElement() { return this.getAriaDescribedByElement(); },

        /**
         * Sets the element for the aria-labelledby attribute. The value of the 'aria-labelledby'
         * attribute is a string id that references another HTML element.  Upon focus, a screen reader should
         * read the content under the HTML element referenced by the 'aria-labelledby' id. Behavior will depend
         * slightly on user agent.
         * @public
         *
         * @param {string} labelId - id referencing the description element
         */
        setAriaLabelledByElement: function( labelElement ) {
          assert && assert( this._domElement, 'HTML element required for aria-labelledby attribute, see setTagName' );
          assert && assert( labelElement.nodeType === ELEMENT_NODE, 'HTML element required' );

          this._ariaLabelledByElement = labelElement;

          this.invalidateAccessibleContent();
        },
        set ariaLabelledByElement( labelElement ) { this.setAriaLabelledByElement( labelElement ); },

        /**
         * Get the element that labels this node's DOM element through the 'aria-labelledby' attribute.
         * See setAriaLabelledBy() for more information about the 'aria-labelledby' attribute behavior.
         * @public
         *
         * @returns {string}
         */
        getAriaLabelledByElement: function() {
          return this._ariaLabelledByElement;
        },
        get ariaLabelledByElement() { return this.getAriaLabelledByElement(); },

        /**
         * If the node is using a list for its description, add a list item to  the end of the list with the text
         * content.  Returns an id so that the element can be referenced if need be.
         * @public
         *
         * @param  {string} textContent
         * @returns {string} - the id of the list item returned for reference
         */
        addDescriptionItem: function( textContent ) {
          assert && assert( this._descriptionElement.tagName === UNORDERED_LIST_TAG, 'description element must be a list to use addDescriptionItem' );

          var listItem = createElement( 'li', false );
          setTextContent( listItem, textContent );

          listItem.id = 'list-item-' + globalListItemCounter++;
          this._descriptionElement.appendChild( listItem );

          return listItem.id;
        },

        /**
         * Update the text content of the description item.  The item may not yet be in the DOM, so
         * document.getElementById cannot be used, see getChildElementWithId()
         * @public
         *
         * @param {string} itemID - id of the lits item to update
         * @param {string} description - new textContent for the string
         */
        updateDescriptionItem: function( itemID, description ) {
          var listItem = getChildElementWithId( this._descriptionElement, itemID );
          assert && assert( this._descriptionElement.tagName === UNORDERED_LIST_TAG, 'description must be a list to hide list items' );
          assert && assert( listItem, 'No list item in description with id ' + itemID );

          setTextContent( listItem, description );
        },

        /**
         * Hide or show the desired list item from the screen reader.
         *
         * @param {string} itemID - id of the list item to hide
         * @param {boolean} hidden - whether the list item should be hidden
         * @public
         */
        setDescriptionItemHidden: function( itemID, hidden ) {
          var listItem = document.getElementById( itemID );
          assert && assert( this._descriptionElement.tagName === UNORDERED_LIST_TAG, 'description must be a list to hide list items' );
          assert && assert( listItem, 'No list item in description with id ' + itemID );

          listItem.hidden = hidden;
        },

        /**
         * Hide completely from a screen reader and the browser by setting the hidden attribute on the node's
         * representative DOM element. If this domElement and its peers have a parent container, the container
         * should be hidden so that all peers are hidden as well.  Hiding the element will remove it from the focus
         * order.
         *
         * @public
         *
         * @param {boolean} hidden
         */
        setAccessibleHidden: function( hidden ) {

          this._accessibleHidden = hidden;
          if ( this._parentContainerElement ) {
            this._parentContainerElement.hidden = hidden;
          }
          else if ( this._domElement ) {
            this._domElement.hidden = hidden;
          }
        },
        set accessibleHidden( hidden ) { this.setAccessibleHidden( hidden ); },

        /**
         * Get whether or not this node's representative DOM element is hidden.
         * @public
         *
         * @returns {boolean}
         */
        getAccessibleHidden: function() {
          return this._accessibleHidden;
        },
        get accessibleHidden() { return this.getAccessibleHidden(); },

        /**
         * Sets whether or not the accessible content should be displayed in the DOM. Almost always, setAccessibleHidden
         * should be used instead of this function.  This should behave exactly like setAccessibleHidden. If removed
         * from display, content will be removed from focus order and undiscoverable with the virtual cursor. Sometimes,
         * hidden attribute is not handled the same way across screen readers, so this function can be used to
         * completely remove the content from the DOM.
         * @public
         *
         * @param {boolean} contentDisplayed
         */
        setAccessibleContentDisplayed: function( contentDisplayed ) {
          this._accessibleContentDisplayed = contentDisplayed;

          for ( var j = 0; j < this._children.length; j++ ) {
            var child = this._children[ j ];
            if ( !contentDisplayed ) {
              child.accessibleContent = null;
            }
            else {
              child.invalidateAccessibleContent();
            }
            child.setAccessibleContentDisplayed( contentDisplayed );
          }
          this.invalidateAccessibleContent();
        },
        set accessibleContentDisplayed( contentDisplayed ) { this.setAccessibleContentDisplayed( contentDisplayed ); },

        getAccessibleContentDisplayed: function() {
          return this._accessibleContentDisplayed;
        },
        get accessibleContentDisplayed() { return this.getAccessibleContentDisplayed(); },

        /**
         * Set the value of an input element.  Element must be a form element to support the value attribute. The input
         * value is converted to string since input values are generally string for HTML.
         * @public
         *
         * @param {string} value
         */
        setInputValue: function( value ) {
          assert && assert( _.includes( FORM_ELEMENTS, this._domElement.tagName ), 'dom element must be a form element to support value' );

          value = value.toString();

          this._inputValue = value;
          this._domElement.value = value;
        },
        set inputValue( value ) { this.setInputValue( value ); },

        /**
         * Get the value of the element. Element must be a form element to support the value attribute.
         * @public
         *
         * @returns {string}
         */
        getInputValue: function() {
          return this._inputValue;
        },
        get inputValue() { return this.getInputValue(); },

        /**
         * Get an array containing all accessible attributes that have been added to this node's DOM element.
         * @public
         *
         * @returns {string[]}
         */
        getAccessibleAttributes: function() {
          return this._accessibleAttributes.slice( 0 ); // defensive copy
        },
        get accessibleAttributes() { return this.getAccessibleAttributes(); },

        /**
         * Set a particular attribute for this node's DOM element, generally to provide extra semantic information for
         * a screen reader.
         *
         * @param {string} attribute - string naming the attribute
         * @param {string|boolean} value - the value for the attribute
         * @public
         */
        setAccessibleAttribute: function( attribute, value ) {
          this._accessibleAttributes.push( { attribute: attribute, value: value } );
          this._domElement.setAttribute( attribute, value );
        },

        /**
         * Remove a particular attribute, removing the associated semantic information from the DOM element.
         *
         * @param {string} attribute - name of the attribute to remove
         * @public
         */
        removeAccessibleAttribute: function( attribute ) {
          assert && assert( elementHasAttribute( this._domElement, attribute ) );
          this._domElement.removeAttribute( attribute );
        },

        /**
         * Remove all attributes from this node's dom element.
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
         * Make the DOM element explicitly focusable with a tab index. Native HTML form elements will generally be in
         * the navigation order without explicitly setting focusable.  If these need to be removed from the navigation
         * order, call setFocusable( false ).  Removing an element from the focus order does not hide the element from
         * assistive technology.
         * @public
         *
         * @param {boolean} isFocusable
         */
        setFocusable: function( isFocusable ) {
          this._focusable = isFocusable;

          if ( this._domElement ) {
            this._domElement.tabIndex = isFocusable ? 0 : -1;
          }
        },
        set focusable( isFocusable ) { this.setFocusable( isFocusable ); },

        /**
         * Get whether or not the node is focusable.
         * @public
         *
         * REVIEW: Usually boolean getters would be called something like isFocusable().
         *
         * @returns {boolean}
         */
        getFocusable: function() {
          return this._focusable;
        },
        get focusable() { return this.getFocusable(); },

        /**
         * Get whether this node's dom element is currently focussed.
         * @public
         *
         * @returns {boolean}
         */
        isFocussed: function() {
          return document.activeElement === this.domElement;
        },
        get focussed() { return this.isFocussed(); },

        /**
         * Focus this node's dom element. The element must not be hidden, and it must be focusable.
         *
         * REVIEW: At call sites, it is not clear that this is related to accessibility.  Consider prepending with
         * 'accessible' or something else to clarify
         * @public
         */
        focus: function() {
          assert && assert( this._domElement.tabIndex !== -1, 'trying to set focus on a node that is not focusable' );
          assert && assert( !this._accessibleHidden, 'trying to set focus on a node with hidden accessible content' );

          // make sure that the element is in the navigation order
          this._domElement.focus();
        },

        /**
         * Remove focus from this DOM element.  The focus highlight will dissapear, and the element will not receive
         * keyboard events when it doesn't have focus.
         * @public
         *
         * REVIEW: At call sites, it is not clear that this is related to accessibility.  Consider prepending with
         * 'accessible' or something else to clarify
         */
        blur: function() {
          this._domElement.blur();
        }

      } );

      /**
       * Returns whether or not the attribute exists on the DOM element.
       *
       * @param  {HTMLElement}  domElement
       * @param  {string}  attribute
       * @returns {string|null}
       */
      function elementHasAttribute( domElement, attribute ) {
        return !!domElement.getAttribute( attribute );
      }

      /**
       * If the text content uses formatting tags, set the content as innerHTML. Otherwise, set as textContent.
       * In general, textContent is more secure and more performant because it doesn't trigger DOM styling and
       * element insertions.
       *
       * @param {HTMLElement} domElement
       * @param {string} textContent
       */
      function setTextContent( domElement, textContent ) {
        if ( textContent && AccessibilityUtil.usesFormattingTagsExclusive( textContent ) ) {

          // XHTML requires <br/> instead of <br>, but <br/> is still valid in HTML. See
          // https://github.com/phetsims/scenery/issues/1309
          var textWithoutBreaks = textContent.replaceAll( '<br>', '<br/>' );
          domElement.innerHTML = textWithoutBreaks;
        }
        else {
          domElement.textContent = textContent;
        }
      }

      /**
       * Get a child element with an id.  This should only be used if the element has not been added to the document
       * yet.  This might happen while setting up or creating the accessible HTML elements. If the element is in the
       * document, document.getElementById is a faster and more conventional option.
       *
       * @param  {HTMLElement} parentElement
       * @param  {string} childId
       * @returns {HTMLElement}
       */
      function getChildElementWithId( parentElement, childId ) {
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

      /**
       * Returns whether or not the element supports innerHTML.
       * @private
       * @param {HTMLElement} domElement
       * @returns {boolean}
       */
      function elementSupportsInnerHTML( domElement ) {
        return !_.includes( ELEMENTS_WITHOUT_CLOSING_TAG, domElement.tagName );
      }

      /**
       * Create an HTML element.  Unless this is a form element or explicitly marked as focusable, add a negative
       * tab index. IE gives all elements a tabIndex of 0 and handles tab navigation internally, so this marks
       * which elements should not be in the focus order.
       *
       * @param  {string} tagName
       * @param {boolean} focusable - should the element be explicitly added to the focus order?
       * @returns {HTMLElement} [description]
       */
      function createElement( tagName, focusable ) {
        var domElement = document.createElement( tagName );

        if ( !_.includes( FORM_ELEMENTS, tagName.toUpperCase() ) && !focusable ) {
          domElement.tabIndex = -1;
        }

        return domElement;
      }

      /**
       * Called by invalidateAccessibleContent.  'this' will be bound by call. The contentElement will either be a
       * label or description element.  The contentElement will be sorted relative to this node's DOM element or its
       * parentContainerElement.  Its placement will also depend on whether or not this node wants to prepend labels,
       * see setPrependLabels().
       * @private
       *
       * @param  {HTMLElement} contentElement
       */
      function insertContentElement( contentElement ) {

        // if we have a parent container, add the element as a child of the container - otherwise, add as child of the
        // node's DOM element
        if ( this._parentContainerElement ) {
          if ( this._prependLabels && this._parentContainerElement === this._domElement.parentNode ) {
            this._parentContainerElement.insertBefore( contentElement, this._domElement );
          }
          else {
            this._parentContainerElement.appendChild( contentElement );
          }
        }
        else {
          this._domElement.appendChild( contentElement );
        }
      }

      /**
       * Some accessibility attributes require a relation to the ID of another element in the DOM. During instantiation,
       * the client can set up a relation to accessible content that is not yet invalidated. If the related element
       * does not have an id, its peer hasn't been created yet, so we flag it with a data attribute to indicate
       * that a relational attribute should be set up via processRelationDataAttributes() when the peer is
       * eventually created.
       *
       * @param {HTMLElement} relatedElement
       * @param {string} attribute
       */
      function addRelationAttribute( relatedElement, attribute ) {
        if ( !relatedElement.id ) {
          relatedElement.setAttribute( DATA_FLAG + attribute, this._accessibleId );
        }
        else {
          this.setAccessibleAttribute( attribute, relatedElement.id );
        }
      }

      /**
       * If an element was flagged with addRelationAttribute above, set the attribute and remove the flag from the
       * DOM element.
       */
      function processRelationDataAttributes() {

        for ( var i = 0; i < RELATION_ATTRIBUTES.length; i++ ) {
          var attribute = RELATION_ATTRIBUTES[ i ];
          var dataAttribute = DATA_FLAG + attribute;

          var relatedElementId = this._domElement.getAttribute( dataAttribute );
          if ( relatedElementId ) {
            var relatedElement = document.getElementById( relatedElementId );

            // set the attribute on the flagged element
            relatedElement.setAttribute( attribute, this.domElement.id );

            // remove the data attribute from this dom element now that it has been processed
            this.domElement.removeAttribute( dataAttribute );
          }
        }
      }

      /**
       * Invalidate our current accessible content, triggering recomputation
       * of anything that depended on the old accessible content. This can be
       * combined with a client implementation of invalidateAccessibleContent.
       *
       * @protected
       */
      function invalidateAccessibleContent() {
        var self = this;

        // clear the parent container if it exists since we will be reinserting labels
        // and the dom element in createPeer
        while ( this._parentContainerElement && this._parentContainerElement.hasChildNodes() ) {
          this._parentContainerElement.removeChild( this._parentContainerElement.lastChild );
        }

        // if any parents are flagged as removed from the accessibility tree, set content to null
        var contentDisplayed = this._accessibleContentDisplayed;
        for ( var i = 0; i < this._parents.length; i++ ) {
          if ( !this._parents[ i ].accessibleContentDisplayed ) {
            contentDisplayed = false;
          }
        }

        var accessibleContent = null;
        if ( contentDisplayed && this._tagName ) {
          accessibleContent = {
            focusHighlight: this._focusHighlight,
            focusHighlightLayerable: this._focusHighlightLayerable,
            createPeer: function( accessibleInstance ) {

              // set up the unique id's for the DOM elements associated with this node's accessible content.
              self._accessibleId = accessibleInstance.trail.getUniqueId();
              self._domElement.id = self._accessibleId;

              // Call this because the _domElement isn't guaranteed to exist before we call createPeer().
              if ( self._accessibleHidden !== null ) {
                self.setAccessibleHidden( self._accessibleHidden );
              }

              // add DOM event listeners to the dom element
              for ( var i = 0; i < self._accessibleInputListenersToAdd.length; i++ ) {
                self.addDOMEventListeners( self._accessibleInputListenersToAdd[ i ] );
              }

              processRelationDataAttributes.call( self );

              // set up id relations for the label element
              if ( self._labelElement ) {
                self._labelElementId = 'label-' + self._accessibleId;
                self._labelElement.id = self._labelElementId;
                if ( self._labelTagName.toUpperCase() === LABEL_TAG ) {
                  self._labelElement.setAttribute( 'for', self._accessibleId );
                }
              }

              // identify the description element
              if ( self._descriptionElement ) {
                self._descriptionElementId = 'description-' + self._accessibleId;
                self._descriptionElement.id = self._descriptionElementId;
              }

              // identify the parent container element
              if ( self._parentContainerElement ) {
                self._parentContainerElement.id = 'parent-container-' + self._accessibleId;

                // provide the aria-role if it is specified
                if ( self._parentContainerAriaRole ) {
                  self._parentContainerElement.setAttribute( 'role', self._parentContainerAriaRole );
                }
              }

              var accessiblePeer = new AccessiblePeer( accessibleInstance, self._domElement, {
                parentContainerElement: self._parentContainerElement
              } );

              // insert the label and description elements in the correct location if they exist
              self._labelElement && insertContentElement.call( self, self._labelElement );
              self._descriptionElement && insertContentElement.call( self, self._descriptionElement );

              // now that the ids have been set up, set up relationships for attributes which depend on the
              // unique ID
              self._ariaDescribedByElement && addRelationAttribute.call( self, self._ariaDescribedByElement, 'aria-describedby' );
              self._ariaLabelledByElement && addRelationAttribute.call( self, self._ariaLabelledByElement, 'aria-labelledby' );

              // Default the focus highlight in this special case to be invisible until selected.
              if ( self._focusHighlightLayerable ) {
                self._focusHighlight.visible = false;
              }

              // if we created a new DOM element, update whether or not it can receive focus
              if ( self._focusable ) {
                self.setFocusable( self._focusable );
              }

              return accessiblePeer;
            }
          };
        }

        this.accessibleContent = accessibleContent;
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