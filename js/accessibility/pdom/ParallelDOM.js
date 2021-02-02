// Copyright 2017-2020, University of Colorado Boulder

/**
 * A trait that is meant to be composed with Node, adding accessibility by defining content for the Parallel DOM.
 *
 * The Parallel DOM is an HTML structure that provides semantics for assistive technologies. For web content to be
 * accessible, assistive technologies require HTML markup, which is something that pure graphical content does not
 * include. This trait adds the accessible HTML content for any Node in the scene graph.
 *
 * Any Node can have pdom content, but they have to opt into it. The structure of the pdom content will
 * match the structure of the scene graph.
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
 * And say that nodes A, B, C, D, and F specify pdom content for the DOM.  Scenery will render the pdom
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
 * node E did not specify pdom content, so node F was added as a child under node C.  If node E had specified
 * pdom content, content for node F would have been added as a child under the content for node E.
 *
 * --------------------------------------------------------------------------------------------------------------------
 * #BASIC EXAMPLE
 *
 * In a basic example let's say that we want to make a Node an unordered list. To do this, add the `tagName` option to
 * the Node, and assign it to the string "ul". Here is what the code could look like:
 *
 * var myUnorderedList = new Node( { tagName: 'ul' } );
 *
 * To get the desired list html, we can assign the `li` `tagName` to children Nodes, like:
 *
 * var listItem1 = new Node( { tagName: 'li' } );
 * myUnorderedList.addChild( listItem1 );
 *
 * Now we have a single list element in the unordered list. To assign content to this <li>, use the `innerContent`
 * option (all of these Node options have getters and setters, just like any other Node option):
 *
 * listItem1.innerContent = 'I am list item number 1';
 *
 * The above operations will create the following PDOM structure (note that actual ids will be different):
 *
 * <ul id='myUnorderedList'>
 *   <li>I am a list item number 1</li>
 * </ul
 *
 * --------------------------------------------------------------------------------------------------------------------
 * #DOM SIBLINGS
 *
 * The api in this trait allows you to add additional structure to the accessible DOM content if necessary. Each node
 * can have multiple DOM Elements associated with it. A Node can have a label DOM element, and a description DOM element.
 * These are called siblings. The Node's direct DOM element (the DOM element you create with the `tagName` option)
 * is called the "primary sibling." You can also have a container parent DOM element that surrounds all of these
 * siblings. With three siblings and a container parent, each Node can have up to 4 DOM Elements representing it in the
 * PDOM. Here is an example of how a Node may use these features:
 *
 * <div>
 *   <label for="myInput">This great label for input</label
 *   <input id="myInput"/>
 *   <p>This is a description for the input</p>
 * </div>
 *
 * Although you can create this structure with four nodes (`input` A, `label B, and `p` C children to `div` D),
 * this structure can be created with one single Node. It is often preferable to do this to limit the number of new
 * Nodes that have to be created just for accessibility purposes. To accomplish this we have the following Node code.
 *
 * new Node( {
 *  tagName: 'input'
 *  labelTagName: 'label',
 *  labelContent: 'This great label for input'
 *  descriptionTagName: 'p',
 *  descriptionContent: 'This is a description for the input',
 *  containerTagName: 'div'
 * });
 *
 * A few notes:
 * 1. Only the primary sibling (specified by tagName) is focusable. Using a focusable element through another element
 *    (like labelTagName) will result in buggy behavior.
 * 2. Notice the names of the content setters for siblings parallel the `innerContent` option for setting the primary
 *    sibling.
 * 3. To make this example actually work, you would need the `inputType` option to set the "type" attribute on the `input`.
 * 4. When you specify the  <label> tag for the label sibling, the "for" attribute is automatically added to the sibling.
 * 5. Finally, the example above doesn't utilize the default tags that we have in place for the parent and siblings.
 *      default labelTagName: 'p'
 *      default descriptionTagName: 'p'
 *      default containerTagName: 'div'
 *    so the following will yield the same PDOM structure:
 *
 *    new Node( {
 *     tagName: 'input',
 *     labelTagName: 'label',
 *     labelContent: 'This great label for input'
 *     descriptionContent: 'This is a description for the input',
 *    });
 *
 * The ParallelDOM trait is smart enough to know when there needs to be a container parent to wrap multiple siblings,
 * it is not necessary to use that option unless the desired tag name is  something other than 'div'.
 *
 * --------------------------------------------------------------------------------------------------------------------
 *
 * For additional accessibility options, please see the options listed in ACCESSIBILITY_OPTION_KEYS. To understand the
 * PDOM more, see PDOMPeer, which manages the DOM Elements for a node. For more documentation on Scenery, Nodes,
 * and the scene graph, please see http://phetsims.github.io/scenery/
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import validate from '../../../../axon/js/validate.js';
import ValidatorDef from '../../../../axon/js/ValidatorDef.js';
import Shape from '../../../../kite/js/Shape.js';
import arrayDifference from '../../../../phet-core/js/arrayDifference.js';
import extend from '../../../../phet-core/js/extend.js';
import merge from '../../../../phet-core/js/merge.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import Trail from '../../util/Trail.js';
import A11yBehaviorFunctionDef from '../A11yBehaviorFunctionDef.js';
import PDOMDisplaysInfo from './PDOMDisplaysInfo.js';
import PDOMInstance from './PDOMInstance.js';
import PDOMPeer from './PDOMPeer.js';
import PDOMTree from './PDOMTree.js';
import PDOMUtils from './PDOMUtils.js';

const INPUT_TAG = PDOMUtils.TAGS.INPUT;
const P_TAG = PDOMUtils.TAGS.P;

// default tag names for siblings
const DEFAULT_DESCRIPTION_TAG_NAME = P_TAG;
const DEFAULT_LABEL_TAG_NAME = P_TAG;

// see setPDOMHeadingBehavior for more details
const DEFAULT_PDOM_HEADING_BEHAVIOR = ( node, options, heading ) => {

  options.labelTagName = 'h' + node.headingLevel; // TODO: make sure heading level change fires a full peer rebuild, see https://github.com/phetsims/scenery/issues/867
  options.labelContent = heading;
  return options;
};

// these elements are typically associated with forms, and support certain attributes
const FORM_ELEMENTS = PDOMUtils.FORM_ELEMENTS;

// list of input "type" attribute values that support the "checked" attribute
const INPUT_TYPES_THAT_SUPPORT_CHECKED = PDOMUtils.INPUT_TYPES_THAT_SUPPORT_CHECKED;

// HTMLElement attributes whose value is an ID of another element
const ASSOCIATION_ATTRIBUTES = PDOMUtils.ASSOCIATION_ATTRIBUTES;

// The options for the ParallelDOM API. In general, most default to null; to clear, set back to null. Each one of
// these has an associated setter, see setter functions for more information about each.
const ACCESSIBILITY_OPTION_KEYS = [

  // Order matters. Having focus before tagName covers the case where you change the tagName and focusability of a
  // currently focused node. We want the focusability to update correctly.
  'focusable', // {boolean|null} - Sets whether or not the node can receive keyboard focus
  'tagName', // {string|null} - Sets the tag name for the primary sibling DOM element in the parallel DOM, should be first

  /*
   * Higher Level API Functions
   */
  'accessibleName', // {string|null} - Sets the name of this node, read when this node receives focus and inserted appropriately based on accessibleNameBehavior
  'accessibleNameBehavior', // {A11yBehaviorFunctionDef} - Sets the way in which accessibleName will be set for the Node, see DEFAULT_ACCESSIBLE_NAME_BEHAVIOR for example
  'helpText', // {string|null} - Sets the descriptive content for this node, read by the virtual cursor, inserted into DOM appropriately based on helpTextBehavior
  'helpTextBehavior', // {A11yBehaviorFunctionDef} - Sets the way in which help text will be set for the Node, see DEFAULT_HELP_TEXT_BEHAVIOR for example
  'pdomHeading', // {string|null} - Sets content for the heading whose level will be automatically generated if specified
  'pdomHeadingBehavior', // {A11yBehaviorFunctionDef} - Set to modify default behavior for inserting pdomHeading string

  /*
   * Lower Level API Functions
   */
  'containerTagName', // {string|null} - Sets the tag name for an [optional] element that contains this Node's siblings
  'containerAriaRole', // {string|null} - Sets the ARIA role for the container parent DOM element

  'innerContent', // {string|null} - Sets the inner text or HTML for a node's primary sibling element
  'inputType', // {string|null} - Sets the input type for the primary sibling DOM element, only relevant if tagName is 'input'
  'inputValue', // {string|null} - Sets the input value for the primary sibling DOM element, only relevant if tagName is 'input'
  'pdomChecked', // {string|null} - Sets the 'checked' state for inputs of type 'radio' and 'checkbox'
  'pdomNamespace', // {string|null} - Sets the namespace for the primary element
  'ariaLabel', // {string|null} - Sets the value of the 'aria-label' attribute on the primary sibling of this Node
  'ariaRole', // {string|null} - Sets the ARIA role for the primary sibling of this Node
  'ariaValueText', // {string|null} - sets the aria-valuetext attribute of the primary sibling

  'labelTagName', // {string|null} - Sets the tag name for the DOM element sibling labeling this node
  'labelContent', // {string|null} - Sets the label content for the node
  'appendLabel', // {string|null} - Sets the label sibling to come after the primary sibling in the PDOM

  'descriptionTagName', // {string|null} - Sets the tag name for the DOM element sibling describing this node
  'descriptionContent', // {string|null} - Sets the description content for the node
  'appendDescription', // {string|null} - Sets the description sibling to come after the primary sibling in the PDOM

  'focusHighlight', // {Node|Shape|null} - Sets the focus highlight for the node
  'focusHighlightLayerable', // {boolean} Flag to determine if the focus highlight node can be layered in the scene graph
  'groupFocusHighlight', // {boolean|Node} - Sets the outer focus highlight for this node when a descendant has focus
  'pdomVisible', // {boolean} - Sets whether or not the node's DOM element is visible in the parallel DOM
  'pdomOrder', // {Array.<Node|null>|null} - Modifies the order of accessible navigation

  'ariaLabelledbyAssociations', // {Array.<Object>} - sets the list of aria-labelledby associations between from this node to others (including itself)
  'ariaDescribedbyAssociations', // {Array.<Object>} - sets the list of aria-describedby associations between from this node to others (including itself)
  'activeDescendantAssociations', // {Array.<Object>} - sets the list of aria-activedescendant associations between from this node to others (including itself)

  'positionInPDOM', // {boolean} - Sets whether or not the node's DOM elements are positioned in the viewport

  'pdomTransformSourceNode' // {Node|null} - sets the node that controls primary sibling element positioning in the display, see setPDOMTransformSourceNode()
];

const ParallelDOM = {

  /**
   * Given the constructor for Node, add accessibility functions into the prototype.
   *
   * @param {function} type - the constructor for Node
   */
  compose( type ) {
    // Can't avoid circular dependency, so no assertion here. Ensure that 'type' is the constructor for Node.
    const proto = type.prototype;

    /**
     * These properties and methods are put directly on the prototype of Node.
     */
    extend( proto, {

      /**
       * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in
       * the order they will be evaluated.  Beware that order matters for accessibility options, changing the order
       * of ACCESSIBILITY_OPTION_KEYS could break the trait.
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
      initializeParallelDOM: function() {

        // @private {string|null} - the HTML tag name of the element representing this node in the DOM
        this._tagName = null;

        // @private {string|null} - the HTML tag name for a container parent element for this node in the DOM. This
        // container parent will contain the node's DOM element, as well as peer elements for any label or description
        // content. See setContainerTagName() for more documentation. If this option is needed (like to
        // contain multiple siblings with the primary sibling), it will default to the value of DEFAULT_CONTAINER_TAG_NAME.
        this._containerTagName = null;

        // @private {string|null} - the HTML tag name for the label element that will contain the label content for
        // this dom element. There are ways in which you can have a label without specifying a label tag name,
        // see setLabelContent() for the list of ways.
        this._labelTagName = null;

        // @private {string|null} - the HTML tag name for the description element that will contain descsription content
        // for this dom element. If a description is set before a tag name is defined, a paragraph element
        // will be created for the description.
        this._descriptionTagName = null;

        // @private {string|null} - the type for an element with tag name of INPUT.  This should only be used
        // if the element has a tag name INPUT.
        this._inputType = null;

        // @private {string|number|null} - the value of the input, only relevant if the tag name is of type "INPUT". Is a
        // string because the `value` attribute is a DOMString. null value indicates no value.
        this._inputValue = null;

        // @private {boolean} - whether or not the pdom input is considered 'checked', only useful for inputs of
        // type 'radio' and 'checkbox'
        this._pdomChecked = false;

        // @private {boolean} - By default the label will be prepended before the primary sibling in the PDOM. This
        // option allows you to instead have the label added after the primary sibling. Note: The label will always
        // be in front of the description sibling. If this flag is set with `appendDescription: true`, the order will be
        // (1) primary sibling, (2) label sibling, (3) description sibling. All siblings will be placed within the
        // containerParent.
        this._appendLabel = false;

        // @private {boolean} - By default the description will be prepended before the primary sibling in the PDOM. This
        // option allows you to instead have the description added after the primary sibling. Note: The description
        // will always be after the label sibling. If this flag is set with `appendLabel: true`, the order will be
        // (1) primary sibling, (2) label sibling, (3) description sibling. All siblings will be placed within the
        // containerParent.
        this._appendDescription = false;

        // @private {Array.<Object> - array of attributes that are on the node's DOM element.  Objects will have the
        // form { attribute:{string}, value:{*}, namespace:{string|null} }
        this._pdomAttributes = [];

        // @private {string|null} - the label content for this node's DOM element.  There are multiple ways that a label
        // can be associated with a node's dom element, see setLabelContent() for more documentation
        this._labelContent = null;

        // @private {string|null} - the inner label content for this node's primary sibling. Set as inner HTML
        // or text content of the actual DOM element. If this is used, the node should not have children.
        this._innerContent = null;

        // @private {string|null} - the description content for this node's DOM element.
        this._descriptionContent = null;

        // @private {string|null} - If provided, it will create the primary DOM element with the specified namespace.
        // This may be needed, for example, with MathML/SVG/etc.
        this._pdomNamespace = null;

        // @private {string|null} - if provided, "aria-label" will be added as an inline attribute on the node's DOM
        // element and set to this value. This will determine how the Accessible Name is provided for the DOM element.
        this._ariaLabel = null;

        // @private {string|null} - the ARIA role for this Node's primary sibling, added as an HTML attribute.  For a complete
        // list of ARIA roles, see https://www.w3.org/TR/wai-aria/roles.  Beware that many roles are not supported
        // by browsers or assistive technologies, so use vanilla HTML for accessibility semantics where possible.
        this._ariaRole = null;

        // @private {string|null} - the ARIA role for the container parent element, added as an HTML attribute. For a
        // complete list of ARIA roles, see https://www.w3.org/TR/wai-aria/roles. Beware that many roles are not
        // supported by browsers or assistive technologies, so use vanilla HTML for accessibility semantics where
        // possible.
        this._containerAriaRole = null;

        // @private {string|null} - if provided, "aria-valuetext" will be added as an inline attribute on the Node's
        // primary sibling and set to this value. Setting back to null will clear this attribute in the view.
        this._ariaValueText = null;

        // @private {Array.<Object>} - Keep track of what this Node is aria-labelledby via "associationObjects"
        // see addAriaLabelledbyAssociation for why we support more than one association.
        this._ariaLabelledbyAssociations = [];

        // Keep a reference to all nodes that are aria-labelledby this node, i.e. that have store one of this Node's
        // peer HTMLElement's id in their peer HTMLElement's aria-labelledby attribute. This way we can tell other
        // nodes to update their aria-labelledby associations when this Node rebuilds its pdom content.
        // @private
        // {Array.<Node>}
        this._nodesThatAreAriaLabelledbyThisNode = [];

        // @private {Array.<Object>} - Keep track of what this Node is aria-describedby via "associationObjects"
        // see addAriaDescribedbyAssociation for why we support more than one association.
        this._ariaDescribedbyAssociations = [];

        // Keep a reference to all nodes that are aria-describedby this node, i.e. that have store one of this Node's
        // peer HTMLElement's id in their peer HTMLElement's aria-describedby attribute. This way we can tell other
        // nodes to update their aria-describedby associations when this Node rebuilds its pdom content.
        // @private
        // {Array.<Node>}
        this._nodesThatAreAriaDescribedbyThisNode = [];

        // @private {Array.<Object>} - Keep track of what this Node is aria-activedescendant via "associationObjects"
        // see addActiveDescendantAssociation for why we support more than one association.
        this._activeDescendantAssociations = [];

        // Keep a reference to all nodes that are aria-activedescendant this node, i.e. that have store one of this Node's
        // peer HTMLElement's id in their peer HTMLElement's aria-activedescendant attribute. This way we can tell other
        // nodes to update their aria-activedescendant associations when this Node rebuilds its pdom content.
        // @private
        // {Array.<Node>}
        this._nodesThatAreActiveDescendantToThisNode = [];

        // @private {boolean|null} - whether or not this Node's primary sibling has been explicitly set to receive focus from
        // tab navigation. Sets the tabIndex attribute on the Node's primary sibling. Setting to false will not remove the
        // node's DOM from the document, but will ensure that it cannot receive focus by pressing 'tab'.  Several
        // HTMLElements (such as HTML form elements) can be focusable by default, without setting this property. The
        // native HTML function from these form elements can be overridden with this property.
        this._focusableOverride = null;

        // @private {Shape|Node|string.<'invisible'>|null} - the focus highlight that will surround this node when it
        // is focused.  By default, the focus highlight will be a pink rectangle that surrounds the Node's local
        // bounds.
        this._focusHighlight = null;

        // @private {boolean} - A flag that allows prevents focus highlight from being displayed in the FocusOverlay.
        // If true, the focus highlight for this node will be layerable in the scene graph.  Client is responsible
        // for placement of the focus highlight in the scene graph.
        this._focusHighlightLayerable = false;

        // @private {boolean|Node} - Adds a group focus highlight that surrounds this node when a descendant has
        // focus. Typically useful to indicate focus if focus enters a group of elements. If 'true', group
        // highlight will go around local bounds of this node. Otherwise the custom node will be used as the highlight/
        this._groupFocusHighlight = false;

        // @private {boolean} - Whether or not the pdom content will be visible from the browser and assistive
        // technologies.  When pdomVisible is false, the Node's primary sibling will not be focusable, and it cannot
        // be found by the assistive technology virtual cursor. For more information on how assistive technologies
        // read with the virtual cursor see
        // http://www.ssbbartgroup.com/blog/how-windows-screen-readers-work-on-the-web/
        this._pdomVisible = true;

        // @private {Array.<Node|null>|null} - (a11y) If provided, it will override the focus order between children
        // (and optionally arbitrary subtrees). If not provided, the focus order will default to the rendering order
        // (first children first, last children last) determined by the children array.
        // See setPDOMOrder() for more documentation.
        this._pdomOrder = null;

        // @public (scenery-internal) {Node|null} - (a11y) If this node is specified in another node's
        // pdomOrder, then this will have the value of that other (PDOM parent) Node. Otherwise it's null.
        this._pdomParent = null;

        // @public (scenery-internal) {Node|null} - If this is specified, the primary sibling will be positioned
        // to align with this source node and observe the transforms along this node's trail. At this time the
        // pdomTransformSourceNode cannot use DAG.
        this._pdomTransformSourceNode = null;

        // @public (scenery-internal) {PDOMDisplaysInfo} - Contains information about what pdom displays
        // this node is "visible" for, see PDOMDisplaysInfo.js for more information.
        this._pdomDisplaysInfo = new PDOMDisplaysInfo( this );

        // @protected {Array.<PDOMInstance>} - Empty unless the Node contains some accessible pdom instance.
        this._pdomInstances = [];

        // @private {boolean} - Determines if DOM siblings are positioned in the viewport. This
        // is required for Nodes that require unique input gestures with iOS VoiceOver like "Drag and Drop".
        // See setPositionInPDOM for more information.
        this._positionInPDOM = false;

        // @public (read-only, scenery-internal) {boolean} - If true, any DOM events received on the label sibling
        // will not dispatch SceneryEvents through the scene graph, see setExcludeLabelSiblingFromInput()
        this.excludeLabelSiblingFromInput = false;

        // HIGHER LEVEL API INITIALIZATION

        // {string|null} - sets the "Accessible Name" of the Node, as defined by the Browser's ParallelDOM Tree
        this._accessibleName = null;

        // {A11yBehaviorFunctionDef} - function that returns the options needed to set the appropriate accessible name for the Node
        this._accessibleNameBehavior = ParallelDOM.BASIC_ACCESSIBLE_NAME_BEHAVIOR;

        // {string|null} - sets the help text of the Node, this most often corresponds to description text.
        this._helpText = null;

        // {A11yBehaviorFunctionDef} - sets the help text of the Node, this most often corresponds to description text.
        this._helpTextBehavior = ParallelDOM.HELP_TEXT_AFTER_CONTENT;

        // {string|null} - sets the help text of the Node, this most often corresponds to label sibling text.
        this._pdomHeading = null;

        // TODO: implement headingLevel override, see https://github.com/phetsims/scenery/issues/855
        // {number|null} - the number that corresponds to the heading tag the node will get if using the pdomHeading api,.
        this._headingLevel = null;

        // {A11yBehaviorFunctionDef} - sets the help text of the Node, this most often corresponds to description text.
        this._pdomHeadingBehavior = DEFAULT_PDOM_HEADING_BEHAVIOR;

        // @private - PDOM specific enabled listener
        this.pdomBoundEnabledListener = this.pdomEnabledListener.bind( this );
        this.enabledProperty.lazyLink( this.pdomBoundEnabledListener );
      },


      /***********************************************************************************************************/
      // PUBLIC METHODS
      /***********************************************************************************************************/

      /**
       * Dispose accessibility by removing all listeners on this node for accessible input. ParallelDOM is disposed
       * by calling Node.dispose(), so this function is scenery-internal.
       * @public (scenery-internal)
       */
      disposeParallelDOM: function() {

        this.enabledProperty.unlink( this.pdomBoundEnabledListener );

        // To prevent memory leaks, we want to clear our order (since otherwise nodes in our order will reference
        // this node).
        this.pdomOrder = null;

        // clear references to the pdomTransformSourceNode
        this.setPDOMTransformSourceNode( null );

        // Clear out aria association attributes, which hold references to other nodes.
        this.setAriaLabelledbyAssociations( [] );
        this.setAriaDescribedbyAssociations( [] );
        this.setActiveDescendantAssociations( [] );
      },

      /**
       * @private
       * @param {boolean} enabled
       */
      pdomEnabledListener: function( enabled ) {

        // Mark this Node as disabled in the ParallelDOM
        this.setPDOMAttribute( 'aria-disabled', !enabled );

        // By returning false, we prevent the component from toggling native HTML element attributes that convey state.
        // For example,this will prevent a checkbox from changing `checked` property while it is disabled. This way
        // we can keep the component in tab order and don't need to add the `disabled` attribute. See
        // https://github.com/phetsims/sun/issues/519 and https://github.com/phetsims/sun/issues/640
        // This solution was found at https://stackoverflow.com/a/12267350/3408502
        this.setPDOMAttribute( 'onclick', enabled ? '' : 'return false' );
      },

      /**
       * Get whether this Node's primary DOM element currently has focus.
       * @public
       *
       * @returns {boolean}
       */
      isFocused: function() {
        for ( let i = 0; i < this._pdomInstances.length; i++ ) {
          const peer = this._pdomInstances[ i ].peer;
          if ( peer.isFocused() ) {
            return true;
          }
        }
        return false;
      },
      get focused() { return this.isFocused(); },

      /**
       * Focus this node's primary dom element. The element must not be hidden, and it must be focusable. If the node
       * has more than one instance, this will fail because the DOM element is not uniquely defined. If accessibility
       * is not enabled, this will be a no op. When ParallelDOM is more widely used, the no op can be replaced
       * with an assertion that checks for pdom content.
       *
       * @public
       */
      focus: function() {

        // if a sim is running without accessibility enabled, there will be no accessible instances, but focus() might
        // still be called without accessibility enabled
        if ( this._pdomInstances.length > 0 ) {

          // when accessibility is widely used, this assertion can be added back in
          // assert && assert( this._pdomInstances.length > 0, 'there must be pdom content for the node to receive focus' );
          assert && assert( this.focusable, 'trying to set focus on a node that is not focusable' );
          assert && assert( this._pdomVisible, 'trying to set focus on a node with invisible pdom content' );
          assert && assert( this._pdomInstances.length === 1, 'focus() unsupported for Nodes using DAG, pdom content is not unique' );

          const peer = this._pdomInstances[ 0 ].peer;
          assert && assert( peer, 'must have a peer to focus' );
          peer.focus();
        }
      },

      /**
       * Remove focus from this node's primary DOM element.  The focus highlight will disappear, and the element will not receive
       * keyboard events when it doesn't have focus.
       * @public
       */
      blur: function() {
        if ( this._pdomInstances.length > 0 ) {
          assert && assert( this._pdomInstances.length === 1, 'blur() unsupported for Nodes using DAG, pdom content is not unique' );
          const peer = this._pdomInstances[ 0 ].peer;
          assert && assert( peer, 'must have a peer to blur' );
          peer.blur();
        }
      },

      /**
       * Called when assertions are enabled and once the Node has been completely constructed. This is the time to
       * make sure that options are saet up the way they are expected to be. For example. you don't want accessibleName
       * and labelContent declared
       * @public (only called by Screen.js)
       */
      pdomAudit: function() {

        if ( this.hasPDOMContent && assert ) {

          this._inputType && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support inputType' );
          this._pdomChecked && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support pdomChecked.' );
          this._inputValue && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support inputValue' );
          this._pdomChecked && assert( INPUT_TYPES_THAT_SUPPORT_CHECKED.indexOf( this._inputType.toUpperCase() ) >= 0, 'inputType does not support checked attribute: ' + this._inputType );
          this._focusHighlightLayerable && assert( this.focusHighlight instanceof Node, 'focusHighlight must be Node if highlight is layerable' );
          this._tagName.toUpperCase() === INPUT_TAG && assert( typeof this._inputType === 'string', ' inputType expected for input' );

          // note that most things that are not focusable by default need innerContent to be focusable on VoiceOver,
          // but this will catch most cases since often things that get added to the focus order have the application
          // role for custom input
          this.ariaRole === 'application' && assert( this._innerContent, 'must have some innerContent or element will never be focusable in VoiceOver' );
        }

        for ( let i = 0; i < this.children.length; i++ ) {
          this.children[ i ].pdomAudit();
        }
      },

      /***********************************************************************************************************/
      // HIGHER LEVEL API: GETTERS AND SETTERS FOR A11Y API OPTIONS
      //
      // These functions utilize the lower level API to achieve a consistence, and convenient API for adding
      // pdom content to the PDOM. See https://github.com/phetsims/scenery/issues/795
      /***********************************************************************************************************/

      /**
       * Set the Node's pdom content in a way that will define the Accessible Name for the browser. Different
       * HTML components and code situations require different methods of setting the Accessible Name. See
       * setAccessibleNameBehavior for details on how this string is rendered in the PDOM. Setting to null will clear
       * this Node's accessibleName
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       *
       * @param {string|null} accessibleName
       */
      setAccessibleName: function( accessibleName ) {
        assert && assert( accessibleName === null || typeof accessibleName === 'string' );

        if ( this._accessibleName !== accessibleName ) {
          this._accessibleName = accessibleName;

          this.onPDOMContentChange();
        }
      },
      set accessibleName( accessibleName ) { this.setAccessibleName( accessibleName ); },

      /**
       * Get the tag name of the DOM element representing this node for accessibility.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       *
       * @returns {string|null}
       */
      getAccessibleName: function() {
        return this._accessibleName;
      },
      get accessibleName() { return this.getAccessibleName(); },

      /**
       * Remove this Node from the PDOM by clearing its pdom content. This can be useful when creating icons from
       * pdom content.
       * @public
       */
      removeFromPDOM: function() {
        assert && assert( this._tagName !== null, 'There is no pdom content to clear from the PDOM' );
        this.tagName = null;
      },


      /**
       * accessibleNameBehavior is a function that will set the appropriate options on this node to get the desired
       * "Accessible Name"
       * @public
       *
       * This accessibleNameBehavior's default does the best it can to create a general method to set the Accessible
       * Name for a variety of different Node types and configurations, but if a Node is more complicated, then this
       * method will not properly set the Accessible Name for the Node's HTML content. In this situation this function
       * needs to be overridden by the subtype to meet its specific constraints. When doing this make it is up to the
       * usage site to make sure that the Accessible Name is properly being set and conveyed to AT, as it is very hard
       * to validate this function.
       *
       * NOTE: By Accessible Name (capitalized), we mean the proper title of the HTML element that will be set in
       * the browser ParallelDOM Tree and then interpreted by AT. This is necessily different from scenery internal
       * names of HTML elements like "label sibling" (even though, in certain circumstances, an Accessible Name could
       * be set by using the "label sibling" with tag name "label" and a "for" attribute).
       *
       * For more information about setting an Accessible Name on HTML see the scenery docs for accessibility,
       * and see https://developer.paciellogroup.com/blog/2017/04/what-is-an-accessible-name/
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @param {A11yBehaviorFunctionDef} accessibleNameBehavior
       */
      setAccessibleNameBehavior: function( accessibleNameBehavior ) {
        assert && A11yBehaviorFunctionDef.validateA11yBehaviorFunctionDef( accessibleNameBehavior );

        if ( this._accessibleNameBehavior !== accessibleNameBehavior ) {

          this._accessibleNameBehavior = accessibleNameBehavior;

          this.onPDOMContentChange();
        }
      },
      set accessibleNameBehavior( accessibleNameBehavior ) { this.setAccessibleNameBehavior( accessibleNameBehavior ); },

      /**
       * Get the help text of the interactive element.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @returns {function}
       */
      getAccessibleNameBehavior: function() {
        return this._accessibleNameBehavior;
      },
      get accessibleNameBehavior() { return this.getAccessibleNameBehavior(); },


      /**
       * Set the Node heading content. This by default will be a heading tag whose level is dependent on how many parents
       * Nodes are heading nodes. See computeHeadingLevel() for more info
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @param {string|null} pdomHeading
       */
      setPDOMHeading: function( pdomHeading ) {
        assert && assert( pdomHeading === null || typeof pdomHeading === 'string' );

        if ( this._pdomHeading !== pdomHeading ) {
          this._pdomHeading = pdomHeading;

          this.onPDOMContentChange();
        }
      },
      set pdomHeading( pdomHeading ) { this.setPDOMHeading( pdomHeading ); },

      /**
       * Get the value of this Node's heading. Use null to clear the heading
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @returns {string|null}
       */
      getPDOMHeading: function() {
        return this._pdomHeading;
      },
      get pdomHeading() { return this.getPDOMHeading(); },


      /**
       * Set the behavior of how `this.pdomHeading` is set in the PDOM. See default behavior function for more
       * information.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @param {A11yBehaviorFunctionDef} pdomHeadingBehavior
       */
      setPDOMHeadingBehavior: function( pdomHeadingBehavior ) {
        assert && A11yBehaviorFunctionDef.validateA11yBehaviorFunctionDef( pdomHeadingBehavior );

        if ( this._pdomHeadingBehavior !== pdomHeadingBehavior ) {

          this._pdomHeadingBehavior = pdomHeadingBehavior;

          this.onPDOMContentChange();
        }
      },
      set pdomHeadingBehavior( pdomHeadingBehavior ) { this.setPDOMHeadingBehavior( pdomHeadingBehavior ); },

      /**
       * Get the help text of the interactive element.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @returns {function}
       */
      getPDOMHeadingBehavior: function() {
        return this._pdomHeadingBehavior;
      },
      get pdomHeadingBehavior() { return this.getPDOMHeadingBehavior(); },


      /**
       * Get the tag name of the DOM element representing this node for accessibility.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @returns {number|null}
       */
      getHeadingLevel: function() {
        return this._headingLevel;
      },
      get headingLevel() { return this.getHeadingLevel(); },


      /**
       // TODO: what if ancestor changes, see https://github.com/phetsims/scenery/issues/855
       * Sets this Node's heading level, by recursing up the accessibility tree to find headings this Node
       * is nested under.
       * @private
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @returns {number}
       */
      computeHeadingLevel: function() {

        // TODO: assert??? assert( this.headingLevel || this._pdomParent); see https://github.com/phetsims/scenery/issues/855
        // Either ^ which may break during construction, or V (below)
        //  base case to heading level 1
        if ( !this._pdomParent ) {
          if ( this._pdomHeading ) {
            this._headingLevel = 1;
            return 1;
          }
          return 0; // so that the first node with a heading is headingLevel 1
        }

        if ( this._pdomHeading ) {
          const level = this._pdomParent.computeHeadingLevel() + 1;
          this._headingLevel = level;
          return level;
        }
        else {
          return this._pdomParent.computeHeadingLevel();
        }
      },

      /**
       * Set the help text for a Node. See setAccessibleNameBehavior for details on how this string is
       * rendered in the PDOM. Null will clear the help text for this Node.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @param {string|null} helpText
       */
      setHelpText: function( helpText ) {
        assert && assert( helpText === null || typeof helpText === 'string' );

        if ( this._helpText !== helpText ) {

          this._helpText = helpText;

          this.onPDOMContentChange();
        }
      },
      set helpText( helpText ) { this.setHelpText( helpText ); },

      /**
       * Get the help text of the interactive element.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @returns {string|null}
       */
      getHelpText: function() {
        return this._helpText;
      },
      get helpText() { return this.getHelpText(); },

      /**
       * helpTextBehavior is a function that will set the appropriate options on this node to get the desired
       * "Help Text".
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @param {A11yBehaviorFunctionDef} helpTextBehavior
       */
      setHelpTextBehavior: function( helpTextBehavior ) {
        assert && A11yBehaviorFunctionDef.validateA11yBehaviorFunctionDef( helpTextBehavior );

        if ( this._helpTextBehavior !== helpTextBehavior ) {

          this._helpTextBehavior = helpTextBehavior;

          this.onPDOMContentChange();
        }
      },
      set helpTextBehavior( helpTextBehavior ) { this.setHelpTextBehavior( helpTextBehavior ); },

      /**
       * Get the help text of the interactive element.
       * @public
       *
       * @experemental - NOTE: use with caution, a11y team reserves the right to change api (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
       * @returns {function}
       */
      getHelpTextBehavior: function() {
        return this._helpTextBehavior;
      },
      get helpTextBehavior() { return this.getHelpTextBehavior(); },


      /***********************************************************************************************************/
      // LOWER LEVEL GETTERS AND SETTERS FOR A11Y API OPTIONS
      /***********************************************************************************************************/

      /**
       * Set the tag name for the primary sibling in the PDOM. DOM element tag names are read-only, so this
       * function will create a new DOM element each time it is called for the Node's PDOMPeer and
       * reset the pdom content.
       * @public
       *
       * @param {string|null} tagName
       */
      setTagName: function( tagName ) {
        assert && assert( tagName === null || typeof tagName === 'string' );

        if ( tagName !== this._tagName ) {
          this._tagName = tagName;

          // TODO: this could be setting a11y content twice
          this.onPDOMContentChange();
        }
      },
      set tagName( tagName ) { this.setTagName( tagName ); },

      /**
       * Get the tag name of the DOM element representing this node for accessibility.
       * @public
       *
       * @returns {string|null}
       */
      getTagName: function() {
        return this._tagName;
      },
      get tagName() { return this.getTagName(); },

      /**
       * Set the tag name for the accessible label sibling for this Node. DOM element tag names are read-only,
       * so this will require creating a new PDOMPeer for this Node (reconstructing all DOM Elements). If
       * labelContent is specified without calling this method, then the DEFAULT_LABEL_TAG_NAME will be used as the
       * tag name for the label sibling.
       * @public
       *
       * Use null to clear the label sibling element from the PDOM.
       *
       * NOTE: This method will create a container parent tagName if none has been specified, because all sibling
       * elements must be children of the container. If you clear the labelTagName and no longer want any
       * content save the primary sibling (this means the container parent as well), then you must manually null out
       * the containerTagName option as well. Although this isn't the greatest strategy, it works for now, and
       * @zepumph and @jessegreenberg can't think of another way to handle this. See for details: https://github.com/phetsims/scenery/issues/761
       *
       *
       * @param {string|null} tagName
       */
      setLabelTagName: function( tagName ) {
        assert && assert( tagName === null || typeof tagName === 'string' );

        if ( tagName !== this._labelTagName ) {
          this._labelTagName = tagName;

          this.onPDOMContentChange();
        }
      },
      set labelTagName( tagName ) { this.setLabelTagName( tagName ); },

      /**
       * Get the label sibling HTML tag name.
       * @public
       *
       * @returns {string|null}
       */
      getLabelTagName: function() {
        return this._labelTagName;
      },
      get labelTagName() { return this.getLabelTagName(); },

      /**
       * Set the tag name for the description sibling. HTML element tag names are read-only, so this will require creating
       * a new HTML element, and inserting it into the DOM. The tag name provided must support
       * innerHTML and textContent. If descriptionContent is specified without this option,
       * then descriptionTagName will be set to DEFAULT_DESCRIPTION_TAG_NAME.
       *
       * Passing 'null' will clear away the description sibling.
       *
       * NOTE: This method will create a container parent tagName if none has been specified. This is because all
       * siblings must be children of the parent container element to appear in the DOM. If you clear
       * the descriptionTagName and no longer want any content other than the primary sibling, you must manually
       * null out the containerTagName option. Although this isn't the greatest strategy, it works for now, and
       * @zepumph and @jessegreenberg can't think of another way to handle this. See for
       * details: https://github.com/phetsims/scenery/issues/761
       *
       * @public
       * @param {string|null} tagName
       */
      setDescriptionTagName: function( tagName ) {
        assert && assert( tagName === null || typeof tagName === 'string' );

        if ( tagName !== this._descriptionTagName ) {

          this._descriptionTagName = tagName;

          this.onPDOMContentChange();
        }
      },
      set descriptionTagName( tagName ) { this.setDescriptionTagName( tagName ); },

      /**
       * Get the HTML tag name for the description sibling.
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
       * specified as readonly, so invalidating pdom content is not necessary.
       * @public
       *
       * @param {string|null} inputType
       */
      setInputType: function( inputType ) {
        assert && assert( inputType === null || typeof inputType === 'string' );
        assert && this.tagName && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tag name must be INPUT to support inputType' );

        if ( inputType !== this._inputType ) {

          this._inputType = inputType;
          for ( let i = 0; i < this._pdomInstances.length; i++ ) {
            const peer = this._pdomInstances[ i ].peer;

            // remove the attribute if cleared by setting to 'null'
            if ( inputType === null ) {
              peer.removeAttributeFromElement( 'type' );
            }
            else {
              peer.setAttributeToElement( 'type', inputType );
            }
          }
        }
      },
      set inputType( inputType ) { this.setInputType( inputType ); },

      /**
       * Get the input type. Input type is only relevant if this Node's primary sibling has tag name "INPUT".
       * @public
       *
       * @returns {string|null}
       */
      getInputType: function() {
        return this._inputType;
      },
      get inputType() { return this.getInputType(); },

      /**
       * By default the label will be prepended before the primary sibling in the PDOM. This
       * option allows you to instead have the label added after the primary sibling. Note: The label will always
       * be in front of the description sibling. If this flag is set with `appendDescription`, the order will be
       * @public
       *
       * <container>
       *   <primary sibling/>
       *   <label sibling/>
       *   <description sibling/>
       * </container>
       * @public
       *
       * @param {boolean} appendLabel
       */
      setAppendLabel: function( appendLabel ) {
        assert && assert( typeof appendLabel === 'boolean' );

        if ( this._appendLabel !== appendLabel ) {
          this._appendLabel = appendLabel;

          this.onPDOMContentChange();
        }
      },
      set appendLabel( appendLabel ) { this.setAppendLabel( appendLabel ); },

      /**
       * Get whether the label sibling should be appended after the primary sibling.
       * @public
       *
       * @returns {boolean}
       */
      getAppendLabel: function() {
        return this._appendLabel;
      },
      get appendLabel() { return this.getAppendLabel(); },

      /**
       * By default the label will be prepended before the primary sibling in the PDOM. This
       * option allows you to instead have the label added after the primary sibling. Note: The label will always
       * be in front of the description sibling. If this flag is set with `appendLabel`, the order will be
       * @public
       *
       * <container>
       *   <primary sibling/>
       *   <label sibling/>
       *   <description sibling/>
       * </container>
       * @public
       *
       * @param {boolean} appendDescription
       */
      setAppendDescription: function( appendDescription ) {
        assert && assert( typeof appendDescription === 'boolean' );

        if ( this._appendDescription !== appendDescription ) {
          this._appendDescription = appendDescription;

          this.onPDOMContentChange();
        }
      },
      set appendDescription( appendDescription ) { this.setAppendDescription( appendDescription ); },

      /**
       * Get whether the description sibling should be appended after the primary sibling.
       * @public
       *
       * @returns {boolean}
       */
      getAppendDescription: function() {
        return this._appendDescription;
      },
      get appendDescription() { return this.getAppendDescription(); },


      /**
       * Set the container parent tag name. By specifying this container parent, an element will be created that
       * acts as a container for this Node's primary sibling DOM Element and its label and description siblings.
       * This containerTagName will default to DEFAULT_LABEL_TAG_NAME, and be added to the PDOM automatically if
       * more than just the primary sibling is created.
       * @public
       *
       * For instance, a button element with a label and description will be contained like the following
       * if the containerTagName is specified as 'section'.
       *
       * <section id='parent-container-trail-id'>
       *   <button>Press me!</button>
       *   <p>Button label</p>
       *   <p>Button description</p>
       * </section>
       *
       * Setting the containerTagName to null directly will result in a no-op if there are still siblings defined for
       * the peer. This is because labelTagName and descriptionTagName will create a parent automatically if one isn't
       * specified. This can result in some weird logic, and @zepumph and @jessegreenberg aren't sure if this is the
       * best way, but it is the way it works for now. See https://github.com/phetsims/scenery/issues/761 for details
       * and if you have opinions to share.
       *
       * @param {string|null} tagName
       */
      setContainerTagName: function( tagName ) {
        assert && assert( tagName === null || typeof tagName === 'string', 'invalid tagName argument: ' + tagName );

        if ( this._containerTagName !== tagName ) {
          this._containerTagName = tagName;
          this.onPDOMContentChange();
        }
      },
      set containerTagName( tagName ) { this.setContainerTagName( tagName ); },

      /**
       * Get the tag name for the container parent element.
       * @public
       *
       * @returns {string|null}
       */
      getContainerTagName: function() {
        return this._containerTagName;
      },
      get containerTagName() { return this.getContainerTagName(); },

      /**
       * Set the content of the label sibling for the this node.  The label sibling will default to the value of
       * DEFAULT_LABEL_TAG_NAME if no `labelTagName` is provided. If the label sibling is a `LABEL` html element,
       * then the `for` attribute will automatically be added, pointing to the Node's primary sibling.
       * @public
       *
       * This method supports adding content in two ways, with HTMLElement.textContent and HTMLElement.innerHTML.
       * The DOM setter is chosen based on if the label passes the `containsFormattingTags`.
       *
       * Passing a null label value will not clear the whole label sibling, just the inner content of the DOM Element.
       * @param {string|null} label
       */
      setLabelContent: function( label ) {
        assert && assert( label === null || typeof label === 'string', 'label must be null or string' );

        if ( this._labelContent !== label ) {
          this._labelContent = label;

          // if trying to set labelContent, make sure that there is a labelTagName default
          if ( !this._labelTagName ) {
            this.setLabelTagName( DEFAULT_LABEL_TAG_NAME );
          }

          for ( let i = 0; i < this._pdomInstances.length; i++ ) {
            const peer = this._pdomInstances[ i ].peer;
            peer.setLabelSiblingContent( this._labelContent );
          }
        }
      },
      set labelContent( label ) { this.setLabelContent( label ); },

      /**
       * Get the content for this Node's label sibling DOM element.
       * @public
       *
       * @returns {string|null}
       */
      getLabelContent: function() {
        return this._labelContent;
      },
      get labelContent() { return this.getLabelContent(); },

      /**
       * Set the inner content for the primary sibling of the AccessiblePeers of this node. Will be set as textContent
       * unless content is html which uses exclusively formatting tags. A node with inner content cannot
       * have accessible descendants because this content will override the HTML of descendants of this node.
       *
       * @param {string|null} content
       * @public
       */
      setInnerContent: function( content ) {
        assert && assert( content === null || typeof content === 'string' );

        if ( this._innerContent !== content ) {
          this._innerContent = content;

          for ( let i = 0; i < this._pdomInstances.length; i++ ) {
            const peer = this._pdomInstances[ i ].peer;
            peer.setPrimarySiblingContent( this._innerContent );
          }
        }
      },
      set innerContent( content ) { this.setInnerContent( content ); },

      /**
       * Get the inner content, the string that is the innerHTML or innerText for the Node's primary sibling.
       *
       * @returns {string|null}
       * @public
       */
      getInnerContent: function() {
        return this._innerContent;
      },
      get innerContent() { return this.getInnerContent(); },

      /**
       * Set the description content for this Node's primary sibling. The description sibling tag name must support
       * innerHTML and textContent. If a description element does not exist yet, a default
       * DEFAULT_LABEL_TAG_NAME will be assigned to the descriptionTagName.
       * @public
       *
       * @param {string|null} descriptionContent
       */
      setDescriptionContent: function( descriptionContent ) {
        assert && assert( descriptionContent === null || typeof descriptionContent === 'string', 'description must be null or string' );

        if ( this._descriptionContent !== descriptionContent ) {
          this._descriptionContent = descriptionContent;

          // if there is no description element, assume that a paragraph element should be used
          if ( !this._descriptionTagName ) {
            this.setDescriptionTagName( DEFAULT_DESCRIPTION_TAG_NAME );
          }

          for ( let i = 0; i < this._pdomInstances.length; i++ ) {
            const peer = this._pdomInstances[ i ].peer;
            peer.setDescriptionSiblingContent( this._descriptionContent );
          }
        }
      },
      set descriptionContent( textContent ) { this.setDescriptionContent( textContent ); },

      /**
       * Get the content for this Node's description sibling DOM Element.
       * @public
       *
       * @returns {string|null}
       */
      getDescriptionContent: function() {
        return this._descriptionContent;
      },
      get descriptionContent() { return this.getDescriptionContent(); },

      /**
       * Set the ARIA role for this Node's primary sibling. According to the W3C, the ARIA role is read-only for a DOM
       * element.  So this will create a new DOM element for this Node with the desired role, and replace the old
       * element in the DOM. Note that the aria role can completely change the events that fire from an element,
       * especially when using a screen reader. For example, a role of `application` will largely bypass the default
       * behavior and logic of the screen reader, triggering keydown/keyup events even for buttons that would usually
       * only receive a "click" event.
       * @public
       *
       * @param {string|null} ariaRole - role for the element, see
       *                            https://www.w3.org/TR/html-aria/#allowed-aria-roles-states-and-properties
       *                            for a list of roles, states, and properties.
       */
      setAriaRole: function( ariaRole ) {
        assert && assert( ariaRole === null || typeof ariaRole === 'string' );

        if ( this._ariaRole !== ariaRole ) {

          this._ariaRole = ariaRole;

          if ( ariaRole !== null ) {
            this.setPDOMAttribute( 'role', ariaRole );
          }
          else {
            this.removePDOMAttribute( 'role' );
          }
        }
      },
      set ariaRole( ariaRole ) { this.setAriaRole( ariaRole ); },

      /**
       * Get the ARIA role representing this node.
       * @public
       *
       * @returns {string|null}
       */
      getAriaRole: function() {
        return this._ariaRole;
      },
      get ariaRole() { return this.getAriaRole(); },

      /**
       * Set the ARIA role for this node's container parent element.  According to the W3C, the ARIA role is read-only
       * for a DOM element. This will create a new DOM element for the container parent with the desired role, and
       * replace it in the DOM.
       * @public
       *
       * @param {string|null} ariaRole - role for the element, see
       *                            https://www.w3.org/TR/html-aria/#allowed-aria-roles-states-and-properties
       *                            for a list of roles, states, and properties.
       */
      setContainerAriaRole: function( ariaRole ) {
        assert && assert( ariaRole === null || typeof ariaRole === 'string' );

        if ( this._containerAriaRole !== ariaRole ) {

          this._containerAriaRole = ariaRole;

          // clear out the attribute
          if ( this._containerAriaRole === null ) {
            this.removePDOMAttribute( 'role', {
              elementName: PDOMPeer.CONTAINER_PARENT
            } );
          }

          // add the attribute
          else {
            this.setPDOMAttribute( 'role', ariaRole, {
              elementName: PDOMPeer.CONTAINER_PARENT
            } );
          }
        }
      },
      set containerAriaRole( ariaRole ) { this.setContainerAriaRole( ariaRole ); },

      /**
       * Get the ARIA role assigned to the container parent element.
       * @public
       * @returns {string|null}
       */
      getContainerAriaRole: function() {
        return this._containerAriaRole;
      },
      get containerAriaRole() { return this.getContainerAriaRole(); },

      /**
       * Set the aria-valuetext of this Node independently from the changing value, if necessary. Setting to null will
       * clear this attribute.
       *
       * @public
       * @param {string|null} ariaValueText
       */
      setAriaValueText: function( ariaValueText ) {
        assert && assert( ariaValueText === null || typeof ariaValueText === 'string' );

        if ( this._ariaValueText !== ariaValueText ) {
          this._ariaValueText = ariaValueText;

          if ( this._ariaValueText === null ) {
            this.removePDOMAttribute( 'aria-valuetext' );
          }
          else {
            this.setPDOMAttribute( 'aria-valuetext', ariaValueText );
          }
        }
      },
      set ariaValueText( ariaValueText ) { this.setAriaValueText( ariaValueText ); },

      /**
       * Get the value of the aria-valuetext attribute for this Node's primary sibling. If null, then the attribute
       * has not been set on the primary sibling.
       * @public
       *
       * @returns {string|null}
       */
      getAriaValueText: function() {
        return this._ariaValueText;
      },
      get ariaValueText() { return this.getAriaValueText(); },

      /**
       * Sets the namespace for the primary element (relevant for MathML/SVG/etc.)
       * @public
       *
       * For example, to create a MathML element:
       * { tagName: 'math', pdomNamespace: 'http://www.w3.org/1998/Math/MathML' }
       *
       * or for SVG:
       * { tagName: 'svg', pdomNamespace: 'http://www.w3.org/2000/svg' }
       *
       * @param {string|null} pdomNamespace - Null indicates no namespace.
       * @returns {Node} - For chaining
       */
      setPDOMNamespace: function( pdomNamespace ) {
        assert && assert( pdomNamespace === null || typeof pdomNamespace === 'string' );

        if ( this._pdomNamespace !== pdomNamespace ) {
          this._pdomNamespace = pdomNamespace;

          // If the namespace changes, tear down the view and redraw the whole thing, there is no easy mutable solution here.
          this.onPDOMContentChange();
        }

        return this;
      },
      set pdomNamespace( value ) { this.setPDOMNamespace( value ); },

      /**
       * Returns the accessible namespace (see setPDOMNamespace for more information).
       * @public
       *
       * @returns {string|null}
       */
      getPDOMNamespace: function() {
        return this._pdomNamespace;
      },
      get pdomNamespace() { return this.getPDOMNamespace(); },

      /**
       * Sets the 'aria-label' attribute for labelling the Node's primary sibling. By using the
       * 'aria-label' attribute, the label will be read on focus, but can not be found with the
       * virtual cursor. This is one way to set a DOM Element's Accessible Name.
       * @public
       *
       * @param {string|null} ariaLabel - the text for the aria label attribute
       */
      setAriaLabel: function( ariaLabel ) {
        assert && assert( ariaLabel === null || typeof ariaLabel === 'string' );

        if ( this._ariaLabel !== ariaLabel ) {
          this._ariaLabel = ariaLabel;

          if ( this._ariaLabel === null ) {
            this.removePDOMAttribute( 'aria-label' );
          }
          else {
            this.setPDOMAttribute( 'aria-label', ariaLabel );
          }
        }
      },
      set ariaLabel( ariaLabel ) { this.setAriaLabel( ariaLabel ); },

      /**
       * Get the value of the aria-label attribute for this Node's primary sibling.
       * @public
       *
       * @returns {string|null}
       */
      getAriaLabel: function() {
        return this._ariaLabel;
      },
      get ariaLabel() { return this.getAriaLabel(); },

      /**
       * Set the focus highlight for this node. By default, the focus highlight will be a pink rectangle that
       * surrounds the node's local bounds.  If focus highlight is set to 'invisible', the node will not have
       * any highlighting when it receives focus.
       * @public
       *
       * @param {Node|Shape|string.<'invisible'>} focusHighlight
       */
      setFocusHighlight: function( focusHighlight ) {
        assert && assert( focusHighlight === null ||
                          focusHighlight instanceof Node ||
                          focusHighlight instanceof Shape ||
                          focusHighlight === 'invisible' );

        if ( this._focusHighlight !== focusHighlight ) {
          this._focusHighlight = focusHighlight;

          // if the focus highlight is layerable in the scene graph, update visibility so that it is only
          // visible when associated node has focus
          if ( this._focusHighlightLayerable ) {

            // if focus highlight is layerable, it must be a node in the scene graph
            assert && assert( focusHighlight instanceof Node );
            focusHighlight.visible = this.focused;
          }

          this.focusHighlightChangedEmitter.emit();
        }
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
       * This will set the visibility of the layered focus highlight, it will always be invisible until this node has
       * focus.
       * @public
       *
       * @param {Boolean} focusHighlightLayerable
       */
      setFocusHighlightLayerable: function( focusHighlightLayerable ) {

        if ( this._focusHighlightLayerable !== focusHighlightLayerable ) {
          this._focusHighlightLayerable = focusHighlightLayerable;

          // if a focus highlight is defined (it must be a node), update its visibility so it is linked to focus
          // of the associated node
          if ( this._focusHighlight ) {
            assert && assert( this._focusHighlight instanceof Node );
            this._focusHighlight.visible = this.focused;
          }
        }
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
       * Set whether or not this node has a group focus highlight. If this node has a group focus highlight, an extra
       * focus highlight will surround this node whenever a descendant node has focus. Generally
       * useful to indicate nested keyboard navigation. If true, the group focus highlight will surround
       * this node's local bounds. Otherwise, the Node will be used.
       *
       * TODO: Support more than one group focus highlight (multiple ancestors could have groupFocusHighlight), see https://github.com/phetsims/scenery/issues/708
       *
       * @public
       * @param {boolean|Node} groupHighlight
       */
      setGroupFocusHighlight: function( groupHighlight ) {
        assert && assert( typeof groupHighlight === 'boolean' || groupHighlight instanceof Node );
        this._groupFocusHighlight = groupHighlight;
      },
      set groupFocusHighlight( groupHighlight ) { this.setGroupFocusHighlight( groupHighlight ); },

      /**
       * Get whether or not this node has a 'group' focus highlight, see setter for more information.
       * @public
       *
       * @returns {Boolean}
       */
      getGroupFocusHighlight: function() {
        return this._groupFocusHighlight;
      },
      get groupFocusHighlight() { return this.getGroupFocusHighlight(); },


      /**
       * Very similar algorithm to setChildren in Node.js
       * @public
       * @param {Array.<Object>} ariaLabelledbyAssociations - list of associationObjects, see this._ariaLabelledbyAssociations.
       */
      setAriaLabelledbyAssociations: function( ariaLabelledbyAssociations ) {
        let associationObject;
        let i;

        // validation if assert is enabled
        if ( assert ) {
          assert( Array.isArray( ariaLabelledbyAssociations ) );
          for ( i = 0; i < ariaLabelledbyAssociations.length; i++ ) {
            associationObject = ariaLabelledbyAssociations[ i ];
            PDOMUtils.validateAssociationObject( associationObject );
          }
        }

        // no work to be done if both are empty, return early
        if ( ariaLabelledbyAssociations.length === 0 && this._ariaLabelledbyAssociations.length === 0 ) {
          return;
        }

        const beforeOnly = []; // Will hold all nodes that will be removed.
        const afterOnly = []; // Will hold all nodes that will be "new" children (added)
        const inBoth = []; // Child nodes that "stay". Will be ordered for the "after" case.

        // get a difference of the desired new list, and the old
        arrayDifference( ariaLabelledbyAssociations, this._ariaLabelledbyAssociations, afterOnly, beforeOnly, inBoth );

        // remove each current associationObject that isn't in the new list
        for ( i = 0; i < beforeOnly.length; i++ ) {
          associationObject = beforeOnly[ i ];
          this.removeAriaLabelledbyAssociation( associationObject );
        }

        assert && assert( this._ariaLabelledbyAssociations.length === inBoth.length,
          'Removing associations should not have triggered other association changes' );

        // add each association from the new list that hasn't been added yet
        for ( i = 0; i < afterOnly.length; i++ ) {
          const ariaLabelledbyAssociation = ariaLabelledbyAssociations[ i ];
          this.addAriaLabelledbyAssociation( ariaLabelledbyAssociation );
        }
      },
      set ariaLabelledbyAssociations( ariaLabelledbyAssociations ) { this.setAriaLabelledbyAssociations( ariaLabelledbyAssociations ); },

      /**
       * @public
       * @returns {Array.<Object>} - the list of current association objects
       */
      getAriaLabelledbyAssociations: function() {
        return this._ariaLabelledbyAssociations;
      },
      get ariaLabelledbyAssociations() { return this.getAriaLabelledbyAssociations(); },

      /**
       * Add an aria-labelledby association to this node. The data in the associationObject will be implemented like
       * "a peer's HTMLElement of this Node (specified with the string constant stored in `thisElementName`) will have an
       * aria-labelledby attribute with a value that includes the `otherNode`'s peer HTMLElement's id (specified with
       * `otherElementName`)."
       * @public)
       *
       * There can be more than one association because an aria-labelledby attribute's value can be a space separated
       * list of HTML ids, and not just a single id, see https://www.w3.org/WAI/GL/wiki/Using_aria-labelledby_to_concatenate_a_label_from_several_text_nodes
       *
       * @param {Object} associationObject - with key value pairs like
       *                               { otherNode: {Node}, otherElementName: {string}, thisElementName: {string } }
       *                               see PDOMPeer for valid element names.
       */
      addAriaLabelledbyAssociation: function( associationObject ) {
        assert && PDOMUtils.validateAssociationObject( associationObject );

        // TODO: assert if this associationObject is already in the association objects list! https://github.com/phetsims/scenery/issues/832

        this._ariaLabelledbyAssociations.push( associationObject ); // Keep track of this association.

        // Flag that this node is is being labelled by the other node, so that if the other node changes it can tell
        // this node to restore the association appropriately.
        associationObject.otherNode._nodesThatAreAriaLabelledbyThisNode.push( this );

        this.updateAriaLabelledbyAssociationsInPeers();
      },

      /**
       * Remove an aria-labelledby association object, see addAriaLabelledbyAssociation for more details
       * @public
       */
      removeAriaLabelledbyAssociation: function( associationObject ) {
        assert && assert( _.includes( this._ariaLabelledbyAssociations, associationObject ) );

        // remove the
        const removedObject = this._ariaLabelledbyAssociations.splice( _.indexOf( this._ariaLabelledbyAssociations, associationObject ), 1 );

        // remove the reference from the other node back to this node because we don't need it anymore
        removedObject[ 0 ].otherNode.removeNodeThatIsAriaLabelledByThisNode( this );

        this.updateAriaLabelledbyAssociationsInPeers();
      },

      /**
       * Remove the reference to the node that is using this Node's ID as an aria-labelledby value
       * @param {Node} node
       * @public (scenery-internal)
       */
      removeNodeThatIsAriaLabelledByThisNode: function( node ) {
        assert && assert( node instanceof Node );
        const indexOfNode = _.indexOf( this._nodesThatAreAriaLabelledbyThisNode, node );
        assert && assert( indexOfNode >= 0 );
        this._nodesThatAreAriaLabelledbyThisNode.splice( indexOfNode, 1 );
      },

      /**
       * Trigger the view update for each PDOMPeer
       * @public
       */
      updateAriaLabelledbyAssociationsInPeers: function() {
        for ( let i = 0; i < this.pdomInstances.length; i++ ) {
          const peer = this.pdomInstances[ i ].peer;
          peer.onAriaLabelledbyAssociationChange();
        }
      },

      /**
       * Update the associations for aria-labelledby
       * @public (scenery-internal)
       */
      updateOtherNodesAriaLabelledby: function() {

        // if any other nodes are aria-labelledby this Node, update those associations too. Since this node's
        // pdom content needs to be recreated, they need to update their aria-labelledby associations accordingly.
        for ( let i = 0; i < this._nodesThatAreAriaLabelledbyThisNode.length; i++ ) {
          const otherNode = this._nodesThatAreAriaLabelledbyThisNode[ i ];
          otherNode.updateAriaLabelledbyAssociationsInPeers();
        }
      },

      /**
       * The list of Nodes that are aria-labelledby this node (other node's peer element will have this Node's Peer element's
       * id in the aria-labelledby attribute
       * @public
       * @returns {Array.<Node>}
       */
      getNodesThatAreAriaLabelledbyThisNode: function() {
        return this._nodesThatAreAriaLabelledbyThisNode;
      },
      get nodesThatAreAriaLabelledbyThisNode() { return this.getNodesThatAreAriaLabelledbyThisNode(); },


      /**
       * @public
       * @param {Array.<Object>} ariaDescribedbyAssociations - list of associationObjects, see this._ariaDescribedbyAssociations.
       */
      setAriaDescribedbyAssociations: function( ariaDescribedbyAssociations ) {
        let associationObject;
        if ( assert ) {
          assert( Array.isArray( ariaDescribedbyAssociations ) );
          for ( let j = 0; j < ariaDescribedbyAssociations.length; j++ ) {
            associationObject = ariaDescribedbyAssociations[ j ];
            assert && PDOMUtils.validateAssociationObject( associationObject );
          }
        }

        // no work to be done if both are empty
        if ( ariaDescribedbyAssociations.length === 0 && this._ariaDescribedbyAssociations.length === 0 ) {
          return;
        }

        const beforeOnly = []; // Will hold all nodes that will be removed.
        const afterOnly = []; // Will hold all nodes that will be "new" children (added)
        const inBoth = []; // Child nodes that "stay". Will be ordered for the "after" case.
        let i;

        // get a difference of the desired new list, and the old
        arrayDifference( ariaDescribedbyAssociations, this._ariaDescribedbyAssociations, afterOnly, beforeOnly, inBoth );

        // remove each current associationObject that isn't in the new list
        for ( i = 0; i < beforeOnly.length; i++ ) {
          associationObject = beforeOnly[ i ];
          this.removeAriaDescribedbyAssociation( associationObject );
        }

        assert && assert( this._ariaDescribedbyAssociations.length === inBoth.length,
          'Removing associations should not have triggered other association changes' );

        // add each association from the new list that hasn't been added yet
        for ( i = 0; i < afterOnly.length; i++ ) {
          const ariaDescribedbyAssociation = ariaDescribedbyAssociations[ i ];
          this.addAriaDescribedbyAssociation( ariaDescribedbyAssociation );
        }
      },
      set ariaDescribedbyAssociations( ariaDescribedbyAssociations ) { this.setAriaDescribedbyAssociations( ariaDescribedbyAssociations ); },

      /**
       * @public
       * @returns {Array.<Object>} - the list of current association objects
       */
      getAriaDescribedbyAssociations: function() {
        return this._ariaDescribedbyAssociations;
      },
      get ariaDescribedbyAssociations() { return this.getAriaDescribedbyAssociations(); },

      /**
       * Add an aria-describedby association to this node. The data in the associationObject will be implemented like
       * "a peer's HTMLElement of this Node (specified with the string constant stored in `thisElementName`) will have an
       * aria-describedby attribute with a value that includes the `otherNode`'s peer HTMLElement's id (specified with
       * `otherElementName`)."
       *
       * There can be more than one association because an aria-describedby attribute's value can be a space separated
       * list of HTML ids, and not just a single id, see https://www.w3.org/WAI/GL/wiki/Using_aria-labelledby_to_concatenate_a_label_from_several_text_nodes
       *
       * @param {Object} associationObject - with key value pairs like
       *                               { otherNode: {Node}, otherElementName: {string}, thisElementName: {string } }
       *                               see PDOMPeer for valid element names.
       */
      addAriaDescribedbyAssociation: function( associationObject ) {
        assert && PDOMUtils.validateAssociationObject( associationObject );
        assert && assert( !_.includes( this._ariaDescribedbyAssociations, associationObject ), 'describedby association already registed' );

        this._ariaDescribedbyAssociations.push( associationObject ); // Keep track of this association.

        // Flag that this node is is being described by the other node, so that if the other node changes it can tell
        // this node to restore the association appropriately.
        associationObject.otherNode._nodesThatAreAriaDescribedbyThisNode.push( this );

        // update the accessiblePeers with this aria-describedby association
        this.updateAriaDescribedbyAssociationsInPeers();
      },

      /**
       * Is this object already in the describedby association list
       * @param {Object} associationObject
       * @returns {boolean}
       */
      hasAriaDescribedbyAssociation: function( associationObject ) {
        return _.includes( this._ariaDescribedbyAssociations, associationObject );
      },

      /**
       * Remove an aria-describedby association object, see addAriaDescribedbyAssociation for more details
       * @public
       */
      removeAriaDescribedbyAssociation: function( associationObject ) {
        assert && assert( _.includes( this._ariaDescribedbyAssociations, associationObject ) );

        // remove the
        const removedObject = this._ariaDescribedbyAssociations.splice( _.indexOf( this._ariaDescribedbyAssociations, associationObject ), 1 );

        // remove the reference from the other node back to this node because we don't need it anymore
        removedObject[ 0 ].otherNode.removeNodeThatIsAriaDescribedByThisNode( this );

        this.updateAriaDescribedbyAssociationsInPeers();
      },

      /**
       * Remove the reference to the node that is using this Node's ID as an aria-describedby value
       * @param {Node} node
       * @public (scenery-internal)
       */
      removeNodeThatIsAriaDescribedByThisNode: function( node ) {
        assert && assert( node instanceof Node );
        const indexOfNode = _.indexOf( this._nodesThatAreAriaDescribedbyThisNode, node );
        assert && assert( indexOfNode >= 0 );
        this._nodesThatAreAriaDescribedbyThisNode.splice( indexOfNode, 1 );

      },

      /**
       * Trigger the view update for each PDOMPeer
       * @public
       */
      updateAriaDescribedbyAssociationsInPeers: function() {
        for ( let i = 0; i < this.pdomInstances.length; i++ ) {
          const peer = this.pdomInstances[ i ].peer;
          peer.onAriaDescribedbyAssociationChange();
        }
      },

      /**
       * Update the associations for aria-describedby
       * @public (scenery-internal)
       */
      updateOtherNodesAriaDescribedby: function() {

        // if any other nodes are aria-describedby this Node, update those associations too. Since this node's
        // pdom content needs to be recreated, they need to update their aria-describedby associations accordingly.
        // TODO: only use unique elements of the array (_.unique)
        for ( let i = 0; i < this._nodesThatAreAriaDescribedbyThisNode.length; i++ ) {
          const otherNode = this._nodesThatAreAriaDescribedbyThisNode[ i ];
          otherNode.updateAriaDescribedbyAssociationsInPeers();
        }
      },

      /**
       * The list of Nodes that are aria-describedby this node (other node's peer element will have this Node's Peer element's
       * id in the aria-describedby attribute
       * @public
       * @returns {Array.<Node>}
       */
      getNodesThatAreAriaDescribedbyThisNode: function() {
        return this._nodesThatAreAriaDescribedbyThisNode;
      },
      get nodesThatAreAriaDescribedbyThisNode() { return this.getNodesThatAreAriaDescribedbyThisNode(); },


      /**
       * @public
       * @param {Array.<Object>} activeDescendantAssociations - list of associationObjects, see this._activeDescendantAssociations.
       */
      setActiveDescendantAssociations: function( activeDescendantAssociations ) {

        let associationObject;
        if ( assert ) {
          assert( Array.isArray( activeDescendantAssociations ) );
          for ( let j = 0; j < activeDescendantAssociations.length; j++ ) {
            associationObject = activeDescendantAssociations[ j ];
            assert && PDOMUtils.validateAssociationObject( associationObject );
          }
        }

        // no work to be done if both are empty, safe to return early
        if ( activeDescendantAssociations.length === 0 && this._activeDescendantAssociations.length === 0 ) {
          return;
        }

        const beforeOnly = []; // Will hold all nodes that will be removed.
        const afterOnly = []; // Will hold all nodes that will be "new" children (added)
        const inBoth = []; // Child nodes that "stay". Will be ordered for the "after" case.
        let i;

        // get a difference of the desired new list, and the old
        arrayDifference( activeDescendantAssociations, this._activeDescendantAssociations, afterOnly, beforeOnly, inBoth );

        // remove each current associationObject that isn't in the new list
        for ( i = 0; i < beforeOnly.length; i++ ) {
          associationObject = beforeOnly[ i ];
          this.removeActiveDescendantAssociation( associationObject );
        }

        assert && assert( this._activeDescendantAssociations.length === inBoth.length,
          'Removing associations should not have triggered other association changes' );

        // add each association from the new list that hasn't been added yet
        for ( i = 0; i < afterOnly.length; i++ ) {
          const activeDescendantAssociation = activeDescendantAssociations[ i ];
          this.addActiveDescendantAssociation( activeDescendantAssociation );
        }
      },
      set activeDescendantAssociations( activeDescendantAssociations ) { this.setActiveDescendantAssociations( activeDescendantAssociations ); },

      /**
       * @public
       * @returns {Array.<Object>} - the list of current association objects
       */
      getActiveDescendantAssociations: function() {
        return this._activeDescendantAssociations;
      },
      get activeDescendantAssociations() { return this.getActiveDescendantAssociations(); },

      /**
       * Add an aria-activeDescendant association to this node. The data in the associationObject will be implemented like
       * "a peer's HTMLElement of this Node (specified with the string constant stored in `thisElementName`) will have an
       * aria-activeDescendant attribute with a value that includes the `otherNode`'s peer HTMLElement's id (specified with
       * `otherElementName`)."
       *
       * @param {Object} associationObject - with key value pairs like
       *                               { otherNode: {Node}, otherElementName: {string}, thisElementName: {string } }
       *                               see PDOMPeer for valid element names.
       */
      addActiveDescendantAssociation: function( associationObject ) {
        assert && PDOMUtils.validateAssociationObject( associationObject );

        // TODO: assert if this associationObject is already in the association objects list! https://github.com/phetsims/scenery/issues/832
        this._activeDescendantAssociations.push( associationObject ); // Keep track of this association.

        // Flag that this node is is being described by the other node, so that if the other node changes it can tell
        // this node to restore the association appropriately.
        associationObject.otherNode._nodesThatAreActiveDescendantToThisNode.push( this );

        // update the accessiblePeers with this aria-activeDescendant association
        this.updateActiveDescendantAssociationsInPeers();
      },

      /**
       * Remove an aria-activeDescendant association object, see addActiveDescendantAssociation for more details
       * @public
       */
      removeActiveDescendantAssociation: function( associationObject ) {
        assert && assert( _.includes( this._activeDescendantAssociations, associationObject ) );

        // remove the
        const removedObject = this._activeDescendantAssociations.splice( _.indexOf( this._activeDescendantAssociations, associationObject ), 1 );

        // remove the reference from the other node back to this node because we don't need it anymore
        removedObject[ 0 ].otherNode.removeNodeThatIsActiveDescendantThisNode( this );

        this.updateActiveDescendantAssociationsInPeers();
      },

      /**
       * Remove the reference to the node that is using this Node's ID as an aria-activeDescendant value
       * @param {Node} node
       * @public (scenery-internal)
       */
      removeNodeThatIsActiveDescendantThisNode: function( node ) {
        assert && assert( node instanceof Node );
        const indexOfNode = _.indexOf( this._nodesThatAreActiveDescendantToThisNode, node );
        assert && assert( indexOfNode >= 0 );
        this._nodesThatAreActiveDescendantToThisNode.splice( indexOfNode, 1 );

      },

      /**
       * Trigger the view update for each PDOMPeer
       * @public
       */
      updateActiveDescendantAssociationsInPeers: function() {
        for ( let i = 0; i < this.pdomInstances.length; i++ ) {
          const peer = this.pdomInstances[ i ].peer;
          peer.onActiveDescendantAssociationChange();
        }
      },

      /**
       * Update the associations for aria-activeDescendant
       * @public (scenery-internal)
       */
      updateOtherNodesActiveDescendant: function() {

        // if any other nodes are aria-activeDescendant this Node, update those associations too. Since this node's
        // pdom content needs to be recreated, they need to update their aria-activeDescendant associations accordingly.
        // TODO: only use unique elements of the array (_.unique)
        for ( let i = 0; i < this._nodesThatAreActiveDescendantToThisNode.length; i++ ) {
          const otherNode = this._nodesThatAreActiveDescendantToThisNode[ i ];
          otherNode.updateActiveDescendantAssociationsInPeers();
        }
      },

      /**
       * The list of Nodes that are aria-activeDescendant this node (other node's peer element will have this Node's Peer element's
       * id in the aria-activeDescendant attribute
       * @public
       * @returns {Array.<Node>}
       */
      getNodesThatAreActiveDescendantToThisNode: function() {
        return this._nodesThatAreActiveDescendantToThisNode;
      },
      get nodesThatAreActiveDescendantToThisNode() { return this.getNodesThatAreActiveDescendantToThisNode(); },


      /**
       * Sets the accessible focus order for this node. This includes not only focused items, but elements that can be
       * placed in the parallel DOM. If provided, it will override the focus order between children (and
       * optionally arbitrary subtrees). If not provided, the focus order will default to the rendering order
       * (first children first, last children last), determined by the children array.
       * @public
       *
       * In the general case, when an accessible order is specified, it's an array of nodes, with optionally one
       * element being a placeholder for "the rest of the children", signified by null. This means that, for
       * accessibility, it will act as if the children for this node WERE the pdomOrder (potentially
       * supplemented with other children via the placeholder).
       *
       * For example, if you have the tree:
       *   a
       *     b
       *       d
       *       e
       *     c
       *       g
       *       f
       *         h
       *
       * and we specify b.pdomOrder = [ e, f, d, c ], then the pdom structure will act as if the tree is:
       *  a
       *    b
       *      e
       *      f <--- the entire subtree of `f` gets placed here under `b`, pulling it out from where it was before.
       *        h
       *      d
       *      c <--- note that `g` is NOT under `c` anymore, because it got pulled out under b directly
       *        g
       *
       * The placeholder (`null`) will get filled in with all direct children that are NOT in any pdomOrder.
       * If there is no placeholder specified, it will act as if the placeholder is at the end of the order.
       * The value `null` (the default) and the empty array (`[]`) both act as if the only order is the placeholder,
       * i.e. `[null]`.
       *
       * Some general constraints for the orders are:
       * - Nodes must be attached to a Display (in a scene graph) to be shown in an accessible order.
       * - You can't specify a node in more than one pdomOrder, and you can't specify duplicates of a value
       *   in an pdomOrder.
       * - You can't specify an ancestor of a node in that node's pdomOrder
       *   (e.g. this.pdomOrder = this.parents ).
       *
       * Note that specifying something in an pdomOrder will effectively remove it from all of its parents for
       * the accessible tree (so if you create `tmpNode.pdomOrder = [ a ]` then toss the tmpNode without
       * disposing it, `a` won't show up in the parallel DOM). If there is a need for that, disposing a Node
       * effectively removes its pdomOrder.
       *
       * See https://github.com/phetsims/scenery-phet/issues/365#issuecomment-381302583 for more information on the
       * decisions and design for this feature.
       *
       * @param {Array.<Node|null>|null} pdomOrder
       */
      setPDOMOrder: function( pdomOrder ) {
        assert && assert( Array.isArray( pdomOrder ) || pdomOrder === null,
          'Array or null expected, received: ' + pdomOrder );
        assert && pdomOrder && pdomOrder.forEach( ( node, index ) => {
          assert( node === null || node instanceof Node,
            'Elements of pdomOrder should be either a Node or null. Element at index ' + index + ' is: ' + node );
        } );
        assert && pdomOrder && assert( this.getTrails( node => _.includes( pdomOrder, node ) ).length === 0, 'pdomOrder should not include any ancestors or the node itself' );

        // Only update if it has changed
        if ( this._pdomOrder !== pdomOrder ) {
          const oldPDOMOrder = this._pdomOrder;

          // Store our own reference to this, so client modifications to the input array won't silently break things.
          // See https://github.com/phetsims/scenery/issues/786
          this._pdomOrder = pdomOrder === null ? null : pdomOrder.slice();

          PDOMTree.pdomOrderChange( this, oldPDOMOrder, pdomOrder );

          this.rendererSummaryRefreshEmitter.emit();
        }
      },
      set pdomOrder( value ) { this.setPDOMOrder( value ); },

      /**
       * Returns the accessible (focus) order for this node.
       * @public
       *
       * @returns {Array.<Node|null>|null}
       */
      getPDOMOrder: function() {
        if ( this._pdomOrder ) {
          return this._pdomOrder.slice( 0 ); // create a defensive copy
        }
        return this._pdomOrder;
      },
      get pdomOrder() { return this.getPDOMOrder(); },

      /**
       * Returns whether this node has an pdomOrder that is effectively different than the default.
       * @public
       *
       * NOTE: `null`, `[]` and `[null]` are all effectively the same thing, so this will return true for any of
       * those. Usage of `null` is recommended, as it doesn't create the extra object reference (but some code
       * that generates arrays may be more convenient).
       *
       * @returns {boolean}
       */
      hasPDOMOrder: function() {
        return this._pdomOrder !== null &&
               this._pdomOrder.length !== 0 &&
               ( this._pdomOrder.length > 1 || this._pdomOrder[ 0 ] !== null );
      },

      /**
       * Returns our "PDOM parent" if available: the node that specifies this node in its pdomOrder.
       * @public
       *
       * @returns {Node|null}
       */
      getPDOMParent: function() {
        return this._pdomParent;
      },
      get pdomParent() { return this.getPDOMParent(); },

      /**
       * Returns the "effective" a11y children for the node (which may be different based on the order or other
       * excluded subtrees).
       * @public
       *
       * If there is no pdomOrder specified, this is basically "all children that don't have accessible panrets"
       * (a Node has a "PDOM parent" if it is specified in an pdomOrder).
       *
       * Otherwise (if it has an pdomOrder), it is the pdomOrder, with the above list of nodes placed
       * in at the location of the placeholder. If there is no placeholder, it acts like a placeholder was the last
       * element of the pdomOrder (see setPDOMOrder for more documentation information).
       *
       * NOTE: If you specify a child in the pdomOrder, it will NOT be double-included (since it will have an
       * PDOM parent).
       *
       * @returns {Array.<Node>}
       */
      getEffectiveChildren: function() {
        // Find all children without PDOM parents.
        const nonOrderedChildren = [];
        for ( let i = 0; i < this._children.length; i++ ) {
          const child = this._children[ i ];

          if ( !child._pdomParent ) {
            nonOrderedChildren.push( child );
          }
        }

        // Override the order, and replace the placeholder if it exists.
        if ( this.hasPDOMOrder() ) {
          const effectiveChildren = this.pdomOrder.slice();

          const placeholderIndex = effectiveChildren.indexOf( null );

          // If we have a placeholder, replace its content with the children
          if ( placeholderIndex >= 0 ) {
            // for efficiency
            nonOrderedChildren.unshift( placeholderIndex, 1 );
            Array.prototype.splice.apply( effectiveChildren, nonOrderedChildren );
          }
          // Otherwise, just add the normal things at the end
          else {
            Array.prototype.push.apply( effectiveChildren, nonOrderedChildren );
          }

          return effectiveChildren;
        }
        else {
          return nonOrderedChildren;
        }
      },

      /**
       * Hide completely from a screen reader and the browser by setting the hidden attribute on the node's
       * representative DOM element. If the sibling DOM Elements have a container parent, the container
       * should be hidden so that all PDOM elements are hidden as well.  Hiding the element will remove it from the focus
       * order.
       *
       * @public
       *
       * @param {boolean} visible
       */
      setPDOMVisible: function( visible ) {
        assert && assert( typeof visible === 'boolean' );
        if ( this._pdomVisible !== visible ) {
          this._pdomVisible = visible;

          this._pdomDisplaysInfo.onPDOMVisibilityChange( visible );
        }
      },
      set pdomVisible( visible ) { this.setPDOMVisible( visible ); },

      /**
       * Get whether or not this node's representative DOM element is visible.
       * @public
       *
       * @returns {boolean}
       */
      isPDOMVisible: function() {
        return this._pdomVisible;
      },
      get pdomVisible() { return this.isPDOMVisible(); },

      /**
       * Returns true if any of the PDOMInstances for the Node are globally visible and displayed in the PDOM. A
       * PDOMInstance is globally visible if Node and all ancestors are pdomVisible. PDOMInstance visibility is
       * updated synchronously, so this returns the most up-to-date information without requiring Display.updateDisplay
       * (unlike Node.wasDisplayed()).
       * @public
       */
      isPDOMDisplayed: function() {
        for ( let i = 0; i < this._pdomInstances.length; i++ ) {
          if ( this._pdomInstances[ i ].isGloballyVisible() ) {
            return true;
          }
        }
        return false;
      },
      get pdomDisplayed() { return this.isPDOMDisplayed(); },

      /**
       * Set the value of an input element.  Element must be a form element to support the value attribute. The input
       * value is converted to string since input values are generally string for HTML.
       * @public
       *
       * @param {string|number} value
       */
      setInputValue: function( value ) {
        assert && assert( value === null || typeof value === 'string' || typeof value === 'number' );
        assert && this._tagName && assert( _.includes( FORM_ELEMENTS, this._tagName.toUpperCase() ), 'dom element must be a form element to support value' );

        // type cast
        value = '' + value;

        if ( value !== this._inputValue ) {
          this._inputValue = value;

          for ( let i = 0; i < this.pdomInstances.length; i++ ) {
            const peer = this.pdomInstances[ i ].peer;
            peer.onInputValueChange();
          }
        }
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
       * Set whether or not the checked attribute appears on the dom elements associated with this Node's
       * pdom content.  This is only useful for inputs of type 'radio' and 'checkbox'. A 'checked' input
       * is considered selected to the browser and assistive technology.
       *
       * @public
       * @param {boolean} checked
       */
      setPDOMChecked: function( checked ) {
        assert && assert( typeof checked === 'boolean' );

        if ( this._tagName ) {
          assert && assert( this._tagName.toUpperCase() === INPUT_TAG, 'Cannot set checked on a non input tag.' );
        }
        if ( this._inputType ) {
          assert && assert( INPUT_TYPES_THAT_SUPPORT_CHECKED.indexOf( this._inputType.toUpperCase() ) >= 0, 'inputType does not support checked: ' + this._inputType );
        }

        if ( this._pdomChecked !== checked ) {
          this._pdomChecked = checked;

          this.setPDOMAttribute( 'checked', checked, {
            asProperty: true
          } );
        }
      },
      set pdomChecked( checked ) { this.setPDOMChecked( checked ); },

      /**
       * Get whether or not the pdom input is 'checked'.
       *
       * @public
       * @returns {boolean}
       */
      getPDOMChecked: function() {
        return this._pdomChecked;
      },
      get pdomChecked() { return this.getPDOMChecked(); },

      /**
       * Get an array containing all pdom attributes that have been added to this Node's primary sibling.
       * @public
       *
       * @returns {Array.<Object>} - Returns objects with: {
       *   attribute: {string} // the name of the attribute
       *   value: {*} // the value of the attribute
       *   options: {options} see options in setPDOMAttribute
       * }
       */
      getPDOMAttributes: function() {
        return this._pdomAttributes.slice( 0 ); // defensive copy
      },
      get pdomAttributes() { return this.getPDOMAttributes(); },

      /**
       * Set a particular attribute or property for this Node's primary sibling, generally to provide extra semantic information for
       * a screen reader.
       *
       * @param {string} attribute - string naming the attribute
       * @param {string|boolean|number} value - the value for the attribute, if boolean, then it will be set as a javascript property on the HTMLElement rather than an attribute
       * @param {Object} [options]
       * @public
       */
      setPDOMAttribute: function( attribute, value, options ) {
        assert && assert( typeof attribute === 'string' );
        assert && assert( typeof value === 'string' || typeof value === 'boolean' || typeof value === 'number' );
        assert && options && assert( Object.getPrototypeOf( options ) === Object.prototype,
          'Extra prototype on pdomAttribute options object is a code smell' );
        assert && typeof value === 'string' && validate( value, ValidatorDef.STRING_WITHOUT_TEMPLATE_VARS_VALIDATOR );

        options = merge( {

          // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
          // for setting certain attributes (e.g. MathML).
          namespace: null,

          // set the "attribute" as a javascript property on the DOMElement instead
          asProperty: false,

          elementName: PDOMPeer.PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
        }, options );

        assert && assert( ASSOCIATION_ATTRIBUTES.indexOf( attribute ) < 0, 'setPDOMAttribute does not support association attributes' );

        // if the pdom attribute already exists in the list, remove it - no need
        // to remove from the peers, existing attributes will simply be replaced in the DOM
        for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
          const currentAttribute = this._pdomAttributes[ i ];
          if ( currentAttribute.attribute === attribute &&
               currentAttribute.options.namespace === options.namespace &&
               currentAttribute.options.asProperty === options.asProperty &&
               currentAttribute.options.elementName === options.elementName ) {
            this._pdomAttributes.splice( i, 1 );
          }
        }

        this._pdomAttributes.push( {
          attribute: attribute,
          value: value,
          options: options
        } );

        for ( let j = 0; j < this._pdomInstances.length; j++ ) {
          const peer = this._pdomInstances[ j ].peer;
          peer.setAttributeToElement( attribute, value, options );
        }
      },

      /**
       * Remove a particular attribute, removing the associated semantic information from the DOM element.
       *
       * @param {string} attribute - name of the attribute to remove
       * @param {Object} [options]
       * @public
       */
      removePDOMAttribute: function( attribute, options ) {
        assert && assert( typeof attribute === 'string' );
        assert && options && assert( Object.getPrototypeOf( options ) === Object.prototype,
          'Extra prototype on pdomAttribute options object is a code smell' );

        options = merge( {

          // {string|null} - If non-null, will remove the attribute with the specified namespace. This can be required
          // for removing certain attributes (e.g. MathML).
          namespace: null,

          elementName: PDOMPeer.PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
        }, options );

        let attributeRemoved = false;
        for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
          if ( this._pdomAttributes[ i ].attribute === attribute &&
               this._pdomAttributes[ i ].options.namespace === options.namespace &&
               this._pdomAttributes[ i ].options.elementName === options.elementName ) {
            this._pdomAttributes.splice( i, 1 );
            attributeRemoved = true;
          }
        }
        assert && assert( attributeRemoved, 'Node does not have pdom attribute ' + attribute );

        for ( let j = 0; j < this._pdomInstances.length; j++ ) {
          const peer = this._pdomInstances[ j ].peer;
          peer.removeAttributeFromElement( attribute, options );
        }
      },

      /**
       * Remove all attributes from this node's dom element.
       * @public
       */
      removePDOMAttributes: function() {

        // all attributes currently on this Node's primary sibling
        const attributes = this.getPDOMAttributes();

        for ( let i = 0; i < attributes.length; i++ ) {
          const attribute = attributes[ i ].attribute;
          this.removePDOMAttribute( attribute );
        }
      },

      /**
       * Remove a particular attribute, removing the associated semantic information from the DOM element.
       *
       * @param {string} attribute - name of the attribute to remove
       * @param {Object} [options]
       * @returns {boolean}
       * @public
       */
      hasPDOMAttribute: function( attribute, options ) {
        assert && assert( typeof attribute === 'string' );
        assert && options && assert( Object.getPrototypeOf( options ) === Object.prototype,
          'Extra prototype on pdomAttribute options object is a code smell' );

        options = merge( {

          // {string|null} - If non-null, will remove the attribute with the specified namespace. This can be required
          // for removing certain attributes (e.g. MathML).
          namespace: null,

          elementName: PDOMPeer.PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
        }, options );

        let attributeFound = false;
        for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
          if ( this._pdomAttributes[ i ].attribute === attribute &&
               this._pdomAttributes[ i ].options.namespace === options.namespace &&
               this._pdomAttributes[ i ].options.elementName === options.elementName ) {
            attributeFound = true;
          }
        }
        return attributeFound;
      },


      /**
       * Make the DOM element explicitly focusable with a tab index. Native HTML form elements will generally be in
       * the navigation order without explicitly setting focusable.  If these need to be removed from the navigation
       * order, call setFocusable( false ).  Removing an element from the focus order does not hide the element from
       * assistive technology.
       * @public
       *
       * @param {boolean|null} focusable - null to use the default browser focus for the primary element
       */
      setFocusable: function( focusable ) {
        assert && assert( focusable === null || typeof focusable === 'boolean' );

        if ( this._focusableOverride !== focusable ) {
          this._focusableOverride = focusable;

          for ( let i = 0; i < this._pdomInstances.length; i++ ) {

            // after the override is set, update the focusability of the peer based on this node's value for focusable
            // which may be true or false (but not null)
            this._pdomInstances[ i ].peer.setFocusable( this.focusable );
          }
        }
      },
      set focusable( isFocusable ) { this.setFocusable( isFocusable ); },

      /**
       * Get whether or not the node is focusable. Use the focusOverride, and then default to browser defined
       * focusable elements.
       * @public
       *
       * @returns {boolean}
       */
      isFocusable: function() {
        if ( this._focusableOverride !== null ) {
          return this._focusableOverride;
        }

        // if there isn't a tagName yet, then there isn't an element, so we aren't focusable. To support option order.
        else if ( this._tagName === null ) {
          return false;
        }
        else {
          return PDOMUtils.tagIsDefaultFocusable( this._tagName );
        }
      },
      get focusable() { return this.isFocusable(); },


      /**
       * Sets the source Node that controls positioning of the primary sibling. Transforms along the trail to this
       * node are observed so that the primary sibling is positioned correctly in the global coordinate frame.
       *
       * The transformSourceNode cannot use DAG for now because we need a unique trail to observe transforms.
       *
       * By default, transforms along trails to all of this Node's PDOMInstances are observed. But this
       * function can be used if you have a visual Node represented in the PDOM by a different Node in the scene
       * graph but still need the other Node's PDOM content positioned over the visual node. For example, this could
       * be required to catch all fake pointer events that may come from certain types of screen readers.
       * @public
       *
       * @param {Node|null} node
       */
      setPDOMTransformSourceNode: function( node ) {
        this._pdomTransformSourceNode = node;

        for ( let i = 0; i < this._pdomInstances.length; i++ ) {
          this._pdomInstances[ i ].peer.setPDOMTransformSourceNode( this._pdomTransformSourceNode );
        }
      },
      set pdomTransformSourceNode( node ) { this.setPDOMTransformSourceNode( node ); },

      /**
       * Get the source Node that controls positioning of the primary sibling in the global coordinate frame. See
       * setPDOMTransformSourceNode for more in depth information.
       * @public
       * @returns {Node|null}
       */
      getPDOMTransformSourceNode: function() {
        return this._pdomTransformSourceNode;
      },
      get pdomTransformSourceNode() { return this.getPDOMTransformSourceNode(); },

      /**
       * Sets whether the PDOM sibling elements are positioned in the correct place in the viewport. Doing so is a
       * requirement for custom gestures on touch based screen readers. However, doing this DOM layout is expensive so
       * only do this when necessary. Generally only needed for elements that utilize a "double tap and hold" gesture
       * to drag and drop.
       *
       * Positioning the PDOM element will caused some screen readers to send both click and pointer events to the
       * location of the Node in global coordinates. Do not position elements that use click listeners since activation
       * will fire twice (once for the pointer event listeners and once for the click event listeners).
       * @public
       *
       * @param {boolean} positionInPDOM
       */
      setPositionInPDOM( positionInPDOM ) {
        this._positionInPDOM = positionInPDOM;

        for ( let i = 0; i < this._pdomInstances.length; i++ ) {
          this._pdomInstances[ i ].peer.setPositionInPDOM( positionInPDOM );
        }
      },
      set positionInPDOM( positionInPDOM ) { this.setPositionInPDOM( positionInPDOM ); },

      /**
       * Gets whether or not we are positioning the PDOM sibling elements. See setPositionInPDOM().
       * @public
       *
       * @returns {boolean}
       */
      getPositionInPDOM() {
        return this._positionInPDOM;
      },
      get positionInPDOM() { return this.getPositionInPDOM(); },

      /**
       * This function should be used sparingly as a workaround. If used, any DOM input events received from the label
       * sibling will not be dispatched as SceneryEvents in Input.js. The label sibling may receive input by screen
       * readers if the virtual cursor is over it. That is usually fine, but there is a bug with NVDA and Firefox where
       * both the label sibling AND primary sibling receive events in this case, and both bubble up to the root of the
       * PDOM, and so we would otherwise dispatch two SceneryEvents instead of one.
       * @public
       *
       * See https://github.com/phetsims/a11y-research/issues/156 for more information.
       */
      setExcludeLabelSiblingFromInput: function() {
        this.excludeLabelSiblingFromInput = true;
        this.onPDOMContentChange();
      },

      /***********************************************************************************************************/
      // SCENERY-INTERNAL AND PRIVATE METHODS
      /***********************************************************************************************************/

      /**
       * Used to get a list of all settable options and their current values.
       * @public (scenery-internal)
       *
       * @returns {Object} - keys are all accessibility option keys, and the values are the values of those properties
       * on this node.
       */
      getBaseOptions: function() {

        const currentOptions = {};

        for ( let i = 0; i < ACCESSIBILITY_OPTION_KEYS.length; i++ ) {
          const optionName = ACCESSIBILITY_OPTION_KEYS[ i ];
          currentOptions[ optionName ] = this[ optionName ];
        }

        return currentOptions;
      },

      /**
       * Returns a recursive data structure that represents the nested ordering of pdom content for this Node's
       * subtree. Each "Item" will have the type { trail: {Trail}, children: {Array.<Item>} }, forming a tree-like
       * structure.
       * @public (scenery-internal)
       *
       * @returns {Array.<Item>}
       */
      getNestedPDOMOrder: function() {
        const currentTrail = new Trail( this );
        let pruneStack = []; // {Array.<Node>} - A list of nodes to prune

        // {Array.<Item>} - The main result we will be returning. It is the top-level array where child items will be
        // inserted.
        const result = [];

        // {Array.<Array.<Item>>} A stack of children arrays, where we should be inserting items into the top array.
        // We will start out with the result, and as nested levels are added, the children arrays of those items will be
        // pushed and poppped, so that the top array on this stack is where we should insert our next child item.
        const nestedChildStack = [ result ];

        function addTrailsForNode( node, overridePruning ) {
          // If subtrees were specified with pdomOrder, they should be skipped from the ordering of ancestor subtrees,
          // otherwise we could end up having multiple references to the same trail (which should be disallowed).
          let pruneCount = 0;
          // count the number of times our node appears in the pruneStack
          _.each( pruneStack, pruneNode => {
            if ( node === pruneNode ) {
              pruneCount++;
            }
          } );

          // If overridePruning is set, we ignore one reference to our node in the prune stack. If there are two copies,
          // however, it means a node was specified in a pdomOrder that already needs to be pruned (so we skip it instead
          // of creating duplicate references in the tab order).
          if ( pruneCount > 1 || ( pruneCount === 1 && !overridePruning ) ) {
            return;
          }

          // Pushing item and its children array, if accessible
          if ( node.hasPDOMContent ) {
            const item = {
              trail: currentTrail.copy(),
              children: []
            };
            nestedChildStack[ nestedChildStack.length - 1 ].push( item );
            nestedChildStack.push( item.children );
          }

          const arrayPDOMOrder = node._pdomOrder === null ? [] : node._pdomOrder;

          // push specific focused nodes to the stack
          pruneStack = pruneStack.concat( arrayPDOMOrder );

          // Visiting trails to ordered nodes.
          _.each( arrayPDOMOrder, descendant => {
            // Find all descendant references to the node.
            // NOTE: We are not reordering trails (due to descendant constraints) if there is more than one instance for
            // this descendant node.
            _.each( node.getLeafTrailsTo( descendant ), descendantTrail => {
              descendantTrail.removeAncestor(); // strip off 'node', so that we handle only children

              // same as the normal order, but adding a full trail (since we may be referencing a descendant node)
              currentTrail.addDescendantTrail( descendantTrail );
              addTrailsForNode( descendant, true ); // 'true' overrides one reference in the prune stack (added above)
              currentTrail.removeDescendantTrail( descendantTrail );
            } );
          } );

          // Visit everything. If there is an pdomOrder, those trails were already visited, and will be excluded.
          const numChildren = node._children.length;
          for ( let i = 0; i < numChildren; i++ ) {
            const child = node._children[ i ];

            currentTrail.addDescendant( child, i );
            addTrailsForNode( child, false );
            currentTrail.removeDescendant();
          }

          // pop focused nodes from the stack (that were added above)
          _.each( arrayPDOMOrder, () => {
            pruneStack.pop();
          } );

          // Popping children array if accessible
          if ( node.hasPDOMContent ) {
            nestedChildStack.pop();
          }
        }

        addTrailsForNode( this, false );

        return result;
      },

      /**
       * Sets the pdom content for a Node. See constructor for more information. Not part of the ParallelDOM
       * API
       * @public (scenery-internal)
       */
      onPDOMContentChange: function() {

        PDOMTree.pdomContentChange( this );

        // recompute the heading level for this node if it is using the pdomHeading API.
        this._pdomHeading && this.computeHeadingLevel();

        this.rendererSummaryRefreshEmitter.emit();
      },

      /**
       * Returns whether or not this Node has any representation for the Parallel DOM.
       * Note this is still true if the content is pdomVisible=false or is otherwise hidden.
       *
       * @public
       *
       * @returns {boolean}
       */
      get hasPDOMContent() {
        return !!this._tagName;
      },

      /**
       * Called when the node is added as a child to this node AND the node's subtree contains pdom content.
       * We need to notify all Displays that can see this change, so that they can update the PDOMInstance tree.
       * @protected (called from Node.js)
       *
       * @param {Node} node
       */
      onPDOMAddChild: function( node ) {
        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( 'onPDOMAddChild n#' + node.id + ' (parent:n#' + this.id + ')' );
        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

        // Find descendants with pdomOrders and check them against all of their ancestors/self
        assert && ( function recur( descendant ) {
          // Prune the search (because milliseconds don't grow on trees, even if we do have assertions enabled)
          if ( descendant._rendererSummary.hasNoPDOM() ) { return; }

          descendant.pdomOrder && assert( descendant.getTrails( node => _.includes( descendant.pdomOrder, node ) ).length === 0, 'pdomOrder should not include any ancestors or the node itself' );
        } )( node );

        assert && PDOMTree.auditNodeForPDOMCycles( this );

        this._pdomDisplaysInfo.onAddChild( node );

        PDOMTree.addChild( this, node );

        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
      },

      /**
       * Called when the node is removed as a child from this node AND the node's subtree contains pdom content.
       * We need to notify all Displays that can see this change, so that they can update the PDOMInstance tree.
       * @private
       *
       * @param {Node} node
       */
      onPDOMRemoveChild: function( node ) {
        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( 'onPDOMRemoveChild n#' + node.id + ' (parent:n#' + this.id + ')' );
        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

        this._pdomDisplaysInfo.onRemoveChild( node );

        PDOMTree.removeChild( this, node );

        // make sure that the associations for aria-labelledby and aria-describedby are updated for nodes associated
        // to this Node (they are pointing to this Node's IDs). https://github.com/phetsims/scenery/issues/816
        node.updateOtherNodesAriaLabelledby();
        node.updateOtherNodesAriaDescribedby();
        node.updateOtherNodesActiveDescendant();

        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
      },

      /**
       * Called when this node's children are reordered (with nothing added/removed).
       * @private
       */
      onPDOMReorderedChildren: function() {
        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( 'onPDOMReorderedChildren (parent:n#' + this.id + ')' );
        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

        PDOMTree.childrenOrderChange( this );

        sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
      },

      /*---------------------------------------------------------------------------*/
      // PDOM Instance handling

      /**
       * Returns a reference to the accessible instances array.
       * @public (scenery-internal)
       *
       * @returns {Array.<PDOMInstance>}
       */
      getPDOMInstances: function() {
        return this._pdomInstances;
      },
      get pdomInstances() { return this.getPDOMInstances(); },

      /**
       * Adds an PDOMInstance reference to our array.
       * @public (scenery-internal)
       *
       * @param {PDOMInstance} pdomInstance
       */
      addPDOMInstance: function( pdomInstance ) {
        assert && assert( pdomInstance instanceof PDOMInstance );
        this._pdomInstances.push( pdomInstance );
      },

      /**
       * Removes an PDOMInstance reference from our array.
       * @public (scenery-internal)
       *
       * @param {PDOMInstance} pdomInstance
       */
      removePDOMInstance: function( pdomInstance ) {
        assert && assert( pdomInstance instanceof PDOMInstance );
        const index = _.indexOf( this._pdomInstances, pdomInstance );
        assert && assert( index !== -1, 'Cannot remove an PDOMInstance from a Node if it was not there' );
        this._pdomInstances.splice( index, 1 );
      }
    } );
  },

  /**
   * @public
   * @type {a11yBehaviorFunction}
   */
  BASIC_ACCESSIBLE_NAME_BEHAVIOR( node, options, accessibleName ) {
    if ( node.tagName === 'input' ) {
      options.labelTagName = 'label';
      options.labelContent = accessibleName;
    }
    else if ( PDOMUtils.tagNameSupportsContent( node.tagName ) ) {
      options.innerContent = accessibleName;
    }
    else {
      options.ariaLabel = accessibleName;
    }
    return options;
  },

  /**
   * @public
   * @type {a11yBehaviorFunction}
   */
  HELP_TEXT_BEFORE_CONTENT( node, options, helpText ) {
    options.descriptionTagName = PDOMUtils.DEFAULT_DESCRIPTION_TAG_NAME;
    options.descriptionContent = helpText;
    options.appendDescription = false;
    return options;
  },

  /**
   * @public
   * @type {a11yBehaviorFunction}
   */
  HELP_TEXT_AFTER_CONTENT( node, options, helpText ) {
    options.descriptionTagName = PDOMUtils.DEFAULT_DESCRIPTION_TAG_NAME;
    options.descriptionContent = helpText;
    options.appendDescription = true;
    return options;
  }
};

scenery.register( 'ParallelDOM', ParallelDOM );
export default ParallelDOM;