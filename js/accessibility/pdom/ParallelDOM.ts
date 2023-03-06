// Copyright 2021-2023, University of Colorado Boulder

/**
 * A superclass for Node, adding accessibility by defining content for the Parallel DOM. Please note that Node and
 * ParallelDOM are closely intertwined, though they are separated into separate files in the type hierarchy.
 *
 * The Parallel DOM is an HTML structure that provides semantics for assistive technologies. For web content to be
 * accessible, assistive technologies require HTML markup, which is something that pure graphical content does not
 * include. This adds the accessible HTML content for any Node in the scene graph.
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
 * The API in this class allows you to add additional structure to the accessible DOM content if necessary. Each node
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
 * The ParallelDOM class is smart enough to know when there needs to be a container parent to wrap multiple siblings,
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

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import validate from '../../../../axon/js/validate.js';
import Validation from '../../../../axon/js/Validation.js';
import { Shape } from '../../../../kite/js/imports.js';
import arrayDifference from '../../../../phet-core/js/arrayDifference.js';
import PhetioObject, { PhetioObjectOptions } from '../../../../tandem/js/PhetioObject.js';
import UtteranceQueue from '../../../../utterance-queue/js/UtteranceQueue.js';
import { TAlertable } from '../../../../utterance-queue/js/Utterance.js';
import { Node, PDOMDisplaysInfo, PDOMInstance, PDOMPeer, PDOMTree, PDOMUtils, scenery, Trail } from '../../imports.js';
import { Highlight } from '../../overlays/HighlightOverlay.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Tandem from '../../../../tandem/js/Tandem.js';
import TEmitter from '../../../../axon/js/TEmitter.js';
import TReadOnlyProperty from '../../../../axon/js/TReadOnlyProperty.js';
import ReadOnlyProperty from '../../../../axon/js/ReadOnlyProperty.js';
import TinyProperty from '../../../../axon/js/TinyProperty.js';
import TinyForwardingProperty from '../../../../axon/js/TinyForwardingProperty.js';
import TProperty from '../../../../axon/js/TProperty.js';

const INPUT_TAG = PDOMUtils.TAGS.INPUT;
const P_TAG = PDOMUtils.TAGS.P;

// default tag names for siblings
const DEFAULT_DESCRIPTION_TAG_NAME = P_TAG;
const DEFAULT_LABEL_TAG_NAME = P_TAG;

export type PDOMValueType = string | TReadOnlyProperty<string>;

// see setPDOMHeadingBehavior for more details
const DEFAULT_PDOM_HEADING_BEHAVIOR = ( node: Node, options: ParallelDOMOptions, heading: PDOMValueType ) => {

  options.labelTagName = `h${node.headingLevel}`; // TODO: make sure heading level change fires a full peer rebuild, see https://github.com/phetsims/scenery/issues/867
  options.labelContent = heading;
  return options;
};

const unwrapProperty = ( valueOrProperty: PDOMValueType | null ): string | null => {
  const result = valueOrProperty === null ? null : ( typeof valueOrProperty === 'string' ? valueOrProperty : valueOrProperty.value );

  assert && assert( result === null || typeof result === 'string' );

  return result;
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
  'focusable',
  'tagName',

  /*
   * Higher Level API Functions
   */
  'accessibleName',
  'accessibleNameBehavior',
  'helpText',
  'helpTextBehavior',
  'pdomHeading',
  'pdomHeadingBehavior',

  /*
   * Lower Level API Functions
   */
  'containerTagName',
  'containerAriaRole',

  'innerContent',
  'inputType',
  'inputValue',
  'pdomChecked',
  'pdomNamespace',
  'ariaLabel',
  'ariaRole',
  'ariaValueText',

  'labelTagName',
  'labelContent',
  'appendLabel',

  'descriptionTagName',
  'descriptionContent',
  'appendDescription',

  'focusHighlight',
  'focusHighlightLayerable',
  'groupFocusHighlight',
  'pdomVisible',
  'pdomOrder',

  'ariaLabelledbyAssociations',
  'ariaDescribedbyAssociations',
  'activeDescendantAssociations',

  'positionInPDOM',

  'pdomTransformSourceNode'
];

// Most options use null for their default behavior, see the setters for each option for a description of how null
// behaves as a default.
export type ParallelDOMOptions = {
  focusable?: boolean | null; // Sets whether the node can receive keyboard focus
  tagName?: string | null; // Sets the tag name for the primary sibling DOM element in the parallel DOM, should be first

  /*
   * Higher Level API Functions
   */
  accessibleName?: PDOMValueType | null; // Sets the name of this node, read when this node receives focus and inserted appropriately based on accessibleNameBehavior
  accessibleNameBehavior?: PDOMBehaviorFunction; // Sets the way in which accessibleName will be set for the Node, see DEFAULT_ACCESSIBLE_NAME_BEHAVIOR for example
  helpText?: PDOMValueType | null; // Sets the descriptive content for this node, read by the virtual cursor, inserted into DOM appropriately based on helpTextBehavior
  helpTextBehavior?: PDOMBehaviorFunction; // Sets the way in which help text will be set for the Node, see DEFAULT_HELP_TEXT_BEHAVIOR for example
  pdomHeading?: PDOMValueType | null; // Sets content for the heading whose level will be automatically generated if specified
  pdomHeadingBehavior?: PDOMBehaviorFunction; // Set to modify default behavior for inserting pdomHeading string

  /*
   * Lower Level API Functions
   */
  containerTagName?: string | null; // Sets the tag name for an [optional] element that contains this Node's siblings
  containerAriaRole?: string | null; // Sets the ARIA role for the container parent DOM element

  innerContent?: PDOMValueType | null; // Sets the inner text or HTML for a node's primary sibling element
  inputType?: string | null; // Sets the input type for the primary sibling DOM element, only relevant if tagName is 'input'
  inputValue?: PDOMValueType | null | number; // Sets the input value for the primary sibling DOM element, only relevant if tagName is 'input'
  pdomChecked?: boolean; // Sets the 'checked' state for inputs of type 'radio' and 'checkbox'
  pdomNamespace?: string | null; // Sets the namespace for the primary element
  ariaLabel?: PDOMValueType | null; // Sets the value of the 'aria-label' attribute on the primary sibling of this Node
  ariaRole?: string | null; // Sets the ARIA role for the primary sibling of this Node
  ariaValueText?: PDOMValueType | null; // sets the aria-valuetext attribute of the primary sibling

  labelTagName?: string | null; // Sets the tag name for the DOM element sibling labeling this node
  labelContent?: PDOMValueType | null; // Sets the label content for the node
  appendLabel?: boolean; // Sets the label sibling to come after the primary sibling in the PDOM

  descriptionTagName?: string | null; // Sets the tag name for the DOM element sibling describing this node
  descriptionContent?: PDOMValueType | null; // Sets the description content for the node
  appendDescription?: boolean; // Sets the description sibling to come after the primary sibling in the PDOM

  focusHighlight?: Highlight; // Sets the focus highlight for the node
  focusHighlightLayerable?: boolean; //lag to determine if the focus highlight node can be layered in the scene graph
  groupFocusHighlight?: Node | boolean; // Sets the outer focus highlight for this node when a descendant has focus
  pdomVisible?: boolean; // Sets whether or not the node's DOM element is visible in the parallel DOM
  pdomOrder?: ( Node | null )[] | null; // Modifies the order of accessible navigation

  ariaLabelledbyAssociations?: Association[]; // sets the list of aria-labelledby associations between from this node to others (including itself)
  ariaDescribedbyAssociations?: Association[]; // sets the list of aria-describedby associations between from this node to others (including itself)
  activeDescendantAssociations?: Association[]; // sets the list of aria-activedescendant associations between from this node to others (including itself)

  positionInPDOM?: boolean; // Sets whether the node's DOM elements are positioned in the viewport

  pdomTransformSourceNode?: Node | null; // { sets the node that controls primary sibling element positioning in the display, see setPDOMTransformSourceNode()
} & PhetioObjectOptions;

type PDOMAttribute = {
  attribute: string;
  value: PDOMValueType | boolean | number;
  namespace: string | null;
  options: SetPDOMAttributeOptions;
};

type PDOMClass = {
  className: string;
  options: SetPDOMClassOptions;
};

type Association = {
  otherNode: Node;
  otherElementName: string;
  thisElementName: string;
};

type SetPDOMAttributeOptions = {
  namespace?: string | null;
  asProperty?: boolean;
  elementName?: string;
};

type RemovePDOMAttributeOptions = {
  namespace?: string | null;
  elementName?: string;
};

type HasPDOMAttributeOptions = {
  namespace?: string | null;
  elementName?: string;
};

type SetPDOMClassOptions = {
  elementName?: string;
};

type RemovePDOMClassOptions = {
  elementName?: string;
};

/**
 *
 * @param node - the node that the pdom behavior is being applied to
 * @param options - options to mutate within the function
 * @param value - the value that you are setting the behavior of, like the accessibleName
 * @param callbacksForOtherNodes - behavior function also support taking state from a Node and using it to
 * set the accessible content for another Node. If this is the case, that logic should be set in a closure and added to
 * this list for execution after this Node is fully created. See discussion in https://github.com/phetsims/sun/issues/503#issuecomment-676541373
 * @returns the options that have been mutated by the behavior function.
 */
export type PDOMBehaviorFunction = ( node: Node, options: ParallelDOMOptions, value: PDOMValueType, callbacksForOtherNodes: ( () => void )[] ) => ParallelDOMOptions;

export default class ParallelDOM extends PhetioObject {

  // The HTML tag name of the element representing this node in the DOM
  private _tagName: string | null;

  // The HTML tag name for a container parent element for this node in the DOM. This
  // container parent will contain the node's DOM element, as well as peer elements for any label or description
  // content. See setContainerTagName() for more documentation. If this option is needed (like to
  // contain multiple siblings with the primary sibling), it will default to the value of DEFAULT_CONTAINER_TAG_NAME.
  private _containerTagName: string | null;

  // The HTML tag name for the label element that will contain the label content for
  // this dom element. There are ways in which you can have a label without specifying a label tag name,
  // see setLabelContent() for the list of ways.
  private _labelTagName: string | null;

  // The HTML tag name for the description element that will contain descsription content
  // for this dom element. If a description is set before a tag name is defined, a paragraph element
  // will be created for the description.
  private _descriptionTagName: string | null;

  // The type for an element with tag name of INPUT.  This should only be used
  // if the element has a tag name INPUT.
  private _inputType: string | null;

  // The value of the input, only relevant if the tag name is of type "INPUT". Is a
  // string because the `value` attribute is a DOMString. null value indicates no value.
  private _inputValue: string | number | null;

  // Whether the pdom input is considered 'checked', only useful for inputs of
  // type 'radio' and 'checkbox'
  private _pdomChecked: boolean;

  // By default the label will be prepended before the primary sibling in the PDOM. This
  // option allows you to instead have the label added after the primary sibling. Note: The label will always
  // be in front of the description sibling. If this flag is set with `appendDescription: true`, the order will be
  // (1) primary sibling, (2) label sibling, (3) description sibling. All siblings will be placed within the
  // containerParent.
  private _appendLabel: boolean;

  // By default the description will be prepended before the primary sibling in the PDOM. This
  // option allows you to instead have the description added after the primary sibling. Note: The description
  // will always be after the label sibling. If this flag is set with `appendLabel: true`, the order will be
  // (1) primary sibling, (2) label sibling, (3) description sibling. All siblings will be placed within the
  // containerParent.
  private _appendDescription: boolean;

  // Array of attributes that are on the node's DOM element.  Objects will have the
  // form { attribute:{string}, value:{*}, namespace:{string|null} }
  private _pdomAttributes: PDOMAttribute[];

  // Collection of class attributes that are applied to the node's DOM element.
  // Objects have the form { className:{string}, options:{*} }
  private _pdomClasses: PDOMClass[];

  // The label content for this node's DOM element.  There are multiple ways that a label
  // can be associated with a node's dom element, see setLabelContent() for more documentation
  private _labelContent: string | null;

  // The inner label content for this node's primary sibling. Set as inner HTML
  // or text content of the actual DOM element. If this is used, the node should not have children.
  private _innerContentProperty: TinyForwardingProperty<string | null>;

  // The description content for this node's DOM element.
  private _descriptionContent: string | null;

  // If provided, it will create the primary DOM element with the specified namespace.
  // This may be needed, for example, with MathML/SVG/etc.
  private _pdomNamespace: string | null;

  // If provided, "aria-label" will be added as an inline attribute on the node's DOM
  // element and set to this value. This will determine how the Accessible Name is provided for the DOM element.
  private _ariaLabel: string | null;

  // The ARIA role for this Node's primary sibling, added as an HTML attribute.  For a complete
  // list of ARIA roles, see https://www.w3.org/TR/wai-aria/roles.  Beware that many roles are not supported
  // by browsers or assistive technologies, so use vanilla HTML for accessibility semantics where possible.
  private _ariaRole: string | null;

  // The ARIA role for the container parent element, added as an HTML attribute. For a
  // complete list of ARIA roles, see https://www.w3.org/TR/wai-aria/roles. Beware that many roles are not
  // supported by browsers or assistive technologies, so use vanilla HTML for accessibility semantics where
  // possible.
  private _containerAriaRole: string | null;

  // If provided, "aria-valuetext" will be added as an inline attribute on the Node's
  // primary sibling and set to this value. Setting back to null will clear this attribute in the view.
  private _ariaValueText: string | null;

  // Keep track of what this Node is aria-labelledby via "associationObjects"
  // see addAriaLabelledbyAssociation for why we support more than one association.
  private _ariaLabelledbyAssociations: Association[];

  // Keep a reference to all nodes that are aria-labelledby this node, i.e. that have store one of this Node's
  // peer HTMLElement's id in their peer HTMLElement's aria-labelledby attribute. This way we can tell other
  // nodes to update their aria-labelledby associations when this Node rebuilds its pdom content.
  private _nodesThatAreAriaLabelledbyThisNode: Node[];

  // Keep track of what this Node is aria-describedby via "associationObjects"
  // see addAriaDescribedbyAssociation for why we support more than one association.
  private _ariaDescribedbyAssociations: Association[];

  // Keep a reference to all nodes that are aria-describedby this node, i.e. that have store one of this Node's
  // peer HTMLElement's id in their peer HTMLElement's aria-describedby attribute. This way we can tell other
  // nodes to update their aria-describedby associations when this Node rebuilds its pdom content.
  private _nodesThatAreAriaDescribedbyThisNode: Node[];

  // Keep track of what this Node is aria-activedescendant via "associationObjects"
  // see addActiveDescendantAssociation for why we support more than one association.
  private _activeDescendantAssociations: Association[];

  // Keep a reference to all nodes that are aria-activedescendant this node, i.e. that have store one of this Node's
  // peer HTMLElement's id in their peer HTMLElement's aria-activedescendant attribute. This way we can tell other
  // nodes to update their aria-activedescendant associations when this Node rebuilds its pdom content.
  private _nodesThatAreActiveDescendantToThisNode: Node[];

  // Whether this Node's primary sibling has been explicitly set to receive focus from
  // tab navigation. Sets the tabIndex attribute on the Node's primary sibling. Setting to false will not remove the
  // node's DOM from the document, but will ensure that it cannot receive focus by pressing 'tab'.  Several
  // HTMLElements (such as HTML form elements) can be focusable by default, without setting this property. The
  // native HTML function from these form elements can be overridden with this property.
  private _focusableOverride: boolean | null;

  // The focus highlight that will surround this node when it
  // is focused.  By default, the focus highlight will be a pink rectangle that surrounds the Node's local
  // bounds.
  private _focusHighlight: Shape | Node | 'invisible' | null;

  // A flag that allows prevents focus highlight from being displayed in the HighlightOverlay.
  // If true, the focus highlight for this node will be layerable in the scene graph.  Client is responsible
  // for placement of the focus highlight in the scene graph.
  private _focusHighlightLayerable: boolean;

  // Adds a group focus highlight that surrounds this node when a descendant has
  // focus. Typically useful to indicate focus if focus enters a group of elements. If 'true', group
  // highlight will go around local bounds of this node. Otherwise the custom node will be used as the highlight/
  private _groupFocusHighlight: Node | boolean;

  // Whether the pdom content will be visible from the browser and assistive
  // technologies.  When pdomVisible is false, the Node's primary sibling will not be focusable, and it cannot
  // be found by the assistive technology virtual cursor. For more information on how assistive technologies
  // read with the virtual cursor see
  // http://www.ssbbartgroup.com/blog/how-windows-screen-readers-work-on-the-web/
  private _pdomVisible: boolean;

  // If provided, it will override the focus order between children
  // (and optionally arbitrary subtrees). If not provided, the focus order will default to the rendering order
  // (first children first, last children last) determined by the children array.
  // See setPDOMOrder() for more documentation.
  private _pdomOrder: ( Node | null )[] | null;

  // If this node is specified in another node's pdomOrder, then this will have the value of that other (PDOM parent)
  // Node. Otherwise it's null.
  // (scenery-internal)
  public _pdomParent: Node | null;

  // If this is specified, the primary sibling will be positioned
  // to align with this source node and observe the transforms along this node's trail. At this time the
  // pdomTransformSourceNode cannot use DAG.
  private _pdomTransformSourceNode: Node | null;

  // Contains information about what pdom displays
  // this node is "visible" for, see PDOMDisplaysInfo.js for more information.
  // (scenery-internal)
  public _pdomDisplaysInfo: PDOMDisplaysInfo;

  // Empty unless the Node contains some pdom content (PDOMInstance).
  private readonly _pdomInstances: PDOMInstance[];

  // Determines if DOM siblings are positioned in the viewport. This
  // is required for Nodes that require unique input gestures with iOS VoiceOver like "Drag and Drop".
  // See setPositionInPDOM for more information.
  private _positionInPDOM: boolean;

  // If true, any DOM events received on the label sibling
  // will not dispatch SceneryEvents through the scene graph, see setExcludeLabelSiblingFromInput() - scenery internal
  private excludeLabelSiblingFromInput: boolean;

  // HIGHER LEVEL API INITIALIZATION

  // Sets the "Accessible Name" of the Node, as defined by the Browser's ParallelDOM Tree
  private _accessibleName: string | null;

  // Function that returns the options needed to set the appropriate accessible name for the Node
  private _accessibleNameBehavior: PDOMBehaviorFunction;

  // Sets the help text of the Node, this most often corresponds to description text.
  private _helpText: string | null;

  // Sets the help text of the Node, this most often corresponds to description text.
  private _helpTextBehavior: PDOMBehaviorFunction;

  // Sets the help text of the Node, this most often corresponds to label sibling text.
  private _pdomHeading: string | null;

  // TODO: implement headingLevel override, see https://github.com/phetsims/scenery/issues/855
  // The number that corresponds to the heading tag the node will get if using the pdomHeading API,.
  private _headingLevel: number | null;

  // Sets the help text of the Node, this most often corresponds to description text.
  private _pdomHeadingBehavior: PDOMBehaviorFunction;

  // Emits an event when the focus highlight is changed.
  public readonly focusHighlightChangedEmitter: TEmitter;

  // Fired when the PDOM Displays for this Node have changed (see PDOMInstance)
  public readonly pdomDisplaysEmitter: TEmitter;

  // PDOM specific enabled listener
  protected pdomBoundInputEnabledListener: ( enabled: boolean ) => void;

  protected constructor( options?: PhetioObjectOptions ) {

    super( options );

    this._tagName = null;
    this._containerTagName = null;
    this._labelTagName = null;
    this._descriptionTagName = null;
    this._inputType = null;
    this._inputValue = null;
    this._pdomChecked = false;
    this._appendLabel = false;
    this._appendDescription = false;
    this._pdomAttributes = [];
    this._pdomClasses = [];
    this._labelContent = null;

    this._innerContentProperty = new TinyForwardingProperty<string | null>( null, false );
    this._innerContentProperty.lazyLink( this.onInnerContentPropertyChange.bind( this ) );

    this._descriptionContent = null;
    this._pdomNamespace = null;
    this._ariaLabel = null;
    this._ariaRole = null;
    this._containerAriaRole = null;
    this._ariaValueText = null;
    this._ariaLabelledbyAssociations = [];
    this._nodesThatAreAriaLabelledbyThisNode = [];
    this._ariaDescribedbyAssociations = [];
    this._nodesThatAreAriaDescribedbyThisNode = [];
    this._activeDescendantAssociations = [];
    this._nodesThatAreActiveDescendantToThisNode = [];
    this._focusableOverride = null;
    this._focusHighlight = null;
    this._focusHighlightLayerable = false;
    this._groupFocusHighlight = false;
    this._pdomVisible = true;
    this._pdomOrder = null;
    this._pdomParent = null;
    this._pdomTransformSourceNode = null;
    this._pdomDisplaysInfo = new PDOMDisplaysInfo( this as unknown as Node );
    this._pdomInstances = [];
    this._positionInPDOM = false;
    this.excludeLabelSiblingFromInput = false;

    // HIGHER LEVEL API INITIALIZATION

    this._accessibleName = null;
    this._accessibleNameBehavior = ParallelDOM.BASIC_ACCESSIBLE_NAME_BEHAVIOR;
    this._helpText = null;
    this._helpTextBehavior = ParallelDOM.HELP_TEXT_AFTER_CONTENT;
    this._pdomHeading = null;
    this._headingLevel = null;
    this._pdomHeadingBehavior = DEFAULT_PDOM_HEADING_BEHAVIOR;
    this.focusHighlightChangedEmitter = new TinyEmitter();
    this.pdomDisplaysEmitter = new TinyEmitter();
    this.pdomBoundInputEnabledListener = this.pdomInputEnabledListener.bind( this );
  }

  /***********************************************************************************************************/
  // PUBLIC METHODS
  /***********************************************************************************************************/

  /**
   * Dispose accessibility by removing all listeners on this node for accessible input. ParallelDOM is disposed
   * by calling Node.dispose(), so this function is scenery-internal.
   * (scenery-internal)
   */
  protected disposeParallelDOM(): void {

    ( this as unknown as Node ).inputEnabledProperty.unlink( this.pdomBoundInputEnabledListener );

    // To prevent memory leaks, we want to clear our order (since otherwise nodes in our order will reference
    // this node).
    this.pdomOrder = null;

    // clear references to the pdomTransformSourceNode
    this.setPDOMTransformSourceNode( null );

    // Clear out aria association attributes, which hold references to other nodes.
    this.setAriaLabelledbyAssociations( [] );
    this.setAriaDescribedbyAssociations( [] );
    this.setActiveDescendantAssociations( [] );

    this._innerContentProperty.dispose();
  }

  private pdomInputEnabledListener( enabled: boolean ): void {

    // Mark this Node as disabled in the ParallelDOM
    this.setPDOMAttribute( 'aria-disabled', !enabled );

    // By returning false, we prevent the component from toggling native HTML element attributes that convey state.
    // For example,this will prevent a checkbox from changing `checked` property while it is disabled. This way
    // we can keep the component in traversal order and don't need to add the `disabled` attribute. See
    // https://github.com/phetsims/sun/issues/519 and https://github.com/phetsims/sun/issues/640
    // This solution was found at https://stackoverflow.com/a/12267350/3408502
    this.setPDOMAttribute( 'onclick', enabled ? '' : 'return false' );
  }

  /**
   * Get whether this Node's primary DOM element currently has focus.
   */
  public isFocused(): boolean {
    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      const peer = this._pdomInstances[ i ].peer!;
      if ( peer.isFocused() ) {
        return true;
      }
    }
    return false;
  }

  public get focused(): boolean { return this.isFocused(); }

  /**
   * Focus this node's primary dom element. The element must not be hidden, and it must be focusable. If the node
   * has more than one instance, this will fail because the DOM element is not uniquely defined. If accessibility
   * is not enabled, this will be a no op. When ParallelDOM is more widely used, the no op can be replaced
   * with an assertion that checks for pdom content.
   */
  public focus(): void {

    // if a sim is running without accessibility enabled, there will be no accessible instances, but focus() might
    // still be called without accessibility enabled
    if ( this._pdomInstances.length > 0 ) {

      // when accessibility is widely used, this assertion can be added back in
      // assert && assert( this._pdomInstances.length > 0, 'there must be pdom content for the node to receive focus' );
      assert && assert( this.focusable, 'trying to set focus on a node that is not focusable' );
      assert && assert( this._pdomVisible, 'trying to set focus on a node with invisible pdom content' );
      assert && assert( this._pdomInstances.length === 1, 'focus() unsupported for Nodes using DAG, pdom content is not unique' );

      const peer = this._pdomInstances[ 0 ].peer!;
      assert && assert( peer, 'must have a peer to focus' );
      peer.focus();
    }
  }

  /**
   * Remove focus from this node's primary DOM element.  The focus highlight will disappear, and the element will not receive
   * keyboard events when it doesn't have focus.
   */
  public blur(): void {
    if ( this._pdomInstances.length > 0 ) {
      assert && assert( this._pdomInstances.length === 1, 'blur() unsupported for Nodes using DAG, pdom content is not unique' );
      const peer = this._pdomInstances[ 0 ].peer!;
      assert && assert( peer, 'must have a peer to blur' );
      peer.blur();
    }
  }

  /**
   * Called when assertions are enabled and once the Node has been completely constructed. This is the time to
   * make sure that options are set up the way they are expected to be. For example. you don't want accessibleName
   * and labelContent declared.
   * (only called by Screen.js)
   */
  public pdomAudit(): void {

    if ( this.hasPDOMContent && assert ) {

      this._inputType && assert( this._tagName!.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support inputType' );
      this._pdomChecked && assert( this._tagName!.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support pdomChecked.' );
      this._inputValue && assert( this._tagName!.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support inputValue' );
      this._pdomChecked && assert( INPUT_TYPES_THAT_SUPPORT_CHECKED.includes( this._inputType!.toUpperCase() ), `inputType does not support checked attribute: ${this._inputType}` );
      this._focusHighlightLayerable && assert( this.focusHighlight instanceof Node, 'focusHighlight must be Node if highlight is layerable' );
      this._tagName!.toUpperCase() === INPUT_TAG && assert( typeof this._inputType === 'string', ' inputType expected for input' );

      // note that most things that are not focusable by default need innerContent to be focusable on VoiceOver,
      // but this will catch most cases since often things that get added to the focus order have the application
      // role for custom input. Note that accessibleName will not be checked that it specifically changes innerContent, it is up to the dev to do this.
      this.ariaRole === 'application' && assert( this._innerContentProperty.value || this._accessibleName, 'must have some innerContent or element will never be focusable in VoiceOver' );
    }

    for ( let i = 0; i < ( this as unknown as Node ).children.length; i++ ) {
      ( this as unknown as Node ).children[ i ].pdomAudit();
    }
  }

  /***********************************************************************************************************/
  // HIGHER LEVEL API: GETTERS AND SETTERS FOR PDOM API OPTIONS
  //
  // These functions utilize the lower level API to achieve a consistence, and convenient API for adding
  // pdom content to the PDOM. See https://github.com/phetsims/scenery/issues/795
  /***********************************************************************************************************/

  /**
   * Set the Node's pdom content in a way that will define the Accessible Name for the browser. Different
   * HTML components and code situations require different methods of setting the Accessible Name. See
   * setAccessibleNameBehavior for details on how this string is rendered in the PDOM. Setting to null will clear
   * this Node's accessibleName
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public setAccessibleName( providedAccessibleName: PDOMValueType | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    const accessibleName = unwrapProperty( providedAccessibleName );

    if ( this._accessibleName !== accessibleName ) {
      this._accessibleName = accessibleName;

      this.onPDOMContentChange();
    }
  }

  public set accessibleName( accessibleName: PDOMValueType | null ) { this.setAccessibleName( accessibleName ); }

  public get accessibleName(): string | null { return this.getAccessibleName(); }

  /**
   * Get the tag name of the DOM element representing this node for accessibility.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely). Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public getAccessibleName(): string | null {
    return this._accessibleName;
  }

  /**
   * Remove this Node from the PDOM by clearing its pdom content. This can be useful when creating icons from
   * pdom content.
   */
  public removeFromPDOM(): void {
    assert && assert( this._tagName !== null, 'There is no pdom content to clear from the PDOM' );
    this.tagName = null;
  }


  /**
   * accessibleNameBehavior is a function that will set the appropriate options on this node to get the desired
   * "Accessible Name"
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
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public setAccessibleNameBehavior( accessibleNameBehavior: PDOMBehaviorFunction ): void {

    if ( this._accessibleNameBehavior !== accessibleNameBehavior ) {

      this._accessibleNameBehavior = accessibleNameBehavior;

      this.onPDOMContentChange();
    }
  }

  public set accessibleNameBehavior( accessibleNameBehavior: PDOMBehaviorFunction ) { this.setAccessibleNameBehavior( accessibleNameBehavior ); }

  public get accessibleNameBehavior(): PDOMBehaviorFunction { return this.getAccessibleNameBehavior(); }

  /**
   * Get the help text of the interactive element.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public getAccessibleNameBehavior(): PDOMBehaviorFunction {
    return this._accessibleNameBehavior;
  }

  /**
   * Set the Node heading content. This by default will be a heading tag whose level is dependent on how many parents
   * Nodes are heading nodes. See computeHeadingLevel() for more info
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public setPDOMHeading( providedPdomHeading: PDOMValueType | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    const pdomHeading = unwrapProperty( providedPdomHeading );

    if ( this._pdomHeading !== pdomHeading ) {
      this._pdomHeading = pdomHeading;

      this.onPDOMContentChange();
    }
  }

  public set pdomHeading( pdomHeading: PDOMValueType | null ) { this.setPDOMHeading( pdomHeading ); }

  public get pdomHeading(): string | null { return this.getPDOMHeading(); }

  /**
   * Get the value of this Node's heading. Use null to clear the heading
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public getPDOMHeading(): string | null {
    return this._pdomHeading;
  }

  /**
   * Set the behavior of how `this.pdomHeading` is set in the PDOM. See default behavior function for more
   * information.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public setPDOMHeadingBehavior( pdomHeadingBehavior: PDOMBehaviorFunction ): void {

    if ( this._pdomHeadingBehavior !== pdomHeadingBehavior ) {

      this._pdomHeadingBehavior = pdomHeadingBehavior;

      this.onPDOMContentChange();
    }
  }

  public set pdomHeadingBehavior( pdomHeadingBehavior: PDOMBehaviorFunction ) { this.setPDOMHeadingBehavior( pdomHeadingBehavior ); }

  public get pdomHeadingBehavior(): PDOMBehaviorFunction { return this.getPDOMHeadingBehavior(); }

  /**
   * Get the help text of the interactive element.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public getPDOMHeadingBehavior(): PDOMBehaviorFunction {
    return this._pdomHeadingBehavior;
  }

  /**
   * Get the tag name of the DOM element representing this node for accessibility.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public getHeadingLevel(): number | null {
    return this._headingLevel;
  }

  public get headingLevel(): number | null { return this.getHeadingLevel(); }


  /**
   // TODO: what if ancestor changes, see https://github.com/phetsims/scenery/issues/855
   * Sets this Node's heading level, by recursing up the accessibility tree to find headings this Node
   * is nested under.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  private computeHeadingLevel(): number {

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
  }

  /**
   * Set the help text for a Node. See setAccessibleNameBehavior for details on how this string is
   * rendered in the PDOM. Null will clear the help text for this Node.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public setHelpText( providedHelpText: PDOMValueType | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    const helpText = unwrapProperty( providedHelpText );

    if ( this._helpText !== helpText ) {

      this._helpText = helpText;

      this.onPDOMContentChange();
    }
  }

  public set helpText( helpText: PDOMValueType | null ) { this.setHelpText( helpText ); }

  public get helpText(): string | null { return this.getHelpText(); }

  /**
   * Get the help text of the interactive element.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public getHelpText(): string | null {
    return this._helpText;
  }

  /**
   * helpTextBehavior is a function that will set the appropriate options on this node to get the desired
   * "Help Text".
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public setHelpTextBehavior( helpTextBehavior: PDOMBehaviorFunction ): void {

    if ( this._helpTextBehavior !== helpTextBehavior ) {

      this._helpTextBehavior = helpTextBehavior;

      this.onPDOMContentChange();
    }
  }

  public set helpTextBehavior( helpTextBehavior: PDOMBehaviorFunction ) { this.setHelpTextBehavior( helpTextBehavior ); }

  public get helpTextBehavior(): PDOMBehaviorFunction { return this.getHelpTextBehavior(); }

  /**
   * Get the help text of the interactive element.
   *
   * @experimental - NOTE: use with caution, a11y team reserves the right to change API (though unlikely).
   *                 Not yet fully implemented, see https://github.com/phetsims/scenery/issues/867
   */
  public getHelpTextBehavior(): PDOMBehaviorFunction {
    return this._helpTextBehavior;
  }


  /***********************************************************************************************************/
  // LOWER LEVEL GETTERS AND SETTERS FOR PDOM API OPTIONS
  /***********************************************************************************************************/

  /**
   * Set the tag name for the primary sibling in the PDOM. DOM element tag names are read-only, so this
   * function will create a new DOM element each time it is called for the Node's PDOMPeer and
   * reset the pdom content.
   *
   * This is the "entry point" for Parallel DOM content. When a Node has a tagName it will appear in the Parallel DOM
   * and other attributes can be set. Without it, nothing will appear in the Parallel DOM.
   */
  public setTagName( tagName: string | null ): void {
    assert && assert( tagName === null || typeof tagName === 'string' );

    if ( tagName !== this._tagName ) {
      this._tagName = tagName;

      // TODO: this could be setting PDOM content twice
      this.onPDOMContentChange();
    }
  }

  public set tagName( tagName: string | null ) { this.setTagName( tagName ); }

  public get tagName(): string | null { return this.getTagName(); }

  /**
   * Get the tag name of the DOM element representing this node for accessibility.
   */
  public getTagName(): string | null {
    return this._tagName;
  }

  /**
   * Set the tag name for the accessible label sibling for this Node. DOM element tag names are read-only,
   * so this will require creating a new PDOMPeer for this Node (reconstructing all DOM Elements). If
   * labelContent is specified without calling this method, then the DEFAULT_LABEL_TAG_NAME will be used as the
   * tag name for the label sibling. Use null to clear the label sibling element from the PDOM.
   */
  public setLabelTagName( tagName: string | null ): void {
    assert && assert( tagName === null || typeof tagName === 'string' );

    if ( tagName !== this._labelTagName ) {
      this._labelTagName = tagName;

      this.onPDOMContentChange();
    }
  }

  public set labelTagName( tagName: string | null ) { this.setLabelTagName( tagName ); }

  public get labelTagName(): string | null { return this.getLabelTagName(); }

  /**
   * Get the label sibling HTML tag name.
   */
  public getLabelTagName(): string | null {
    return this._labelTagName;
  }

  /**
   * Set the tag name for the description sibling. HTML element tag names are read-only, so this will require creating
   * a new HTML element, and inserting it into the DOM. The tag name provided must support
   * innerHTML and textContent. If descriptionContent is specified without this option,
   * then descriptionTagName will be set to DEFAULT_DESCRIPTION_TAG_NAME.
   *
   * Passing 'null' will clear away the description sibling.
   */
  public setDescriptionTagName( tagName: string | null ): void {
    assert && assert( tagName === null || typeof tagName === 'string' );

    if ( tagName !== this._descriptionTagName ) {

      this._descriptionTagName = tagName;

      this.onPDOMContentChange();
    }
  }

  public set descriptionTagName( tagName: string | null ) { this.setDescriptionTagName( tagName ); }

  public get descriptionTagName(): string | null { return this.getDescriptionTagName(); }

  /**
   * Get the HTML tag name for the description sibling.
   */
  public getDescriptionTagName(): string | null {
    return this._descriptionTagName;
  }

  /**
   * Sets the type for an input element.  Element must have the INPUT tag name. The input attribute is not
   * specified as readonly, so invalidating pdom content is not necessary.
   */
  public setInputType( inputType: string | null ): void {
    assert && assert( inputType === null || typeof inputType === 'string' );
    assert && this.tagName && assert( this._tagName!.toUpperCase() === INPUT_TAG, 'tag name must be INPUT to support inputType' );

    if ( inputType !== this._inputType ) {

      this._inputType = inputType;
      for ( let i = 0; i < this._pdomInstances.length; i++ ) {
        const peer = this._pdomInstances[ i ].peer!;

        // remove the attribute if cleared by setting to 'null'
        if ( inputType === null ) {
          peer.removeAttributeFromElement( 'type' );
        }
        else {
          peer.setAttributeToElement( 'type', inputType );
        }
      }
    }
  }

  public set inputType( inputType: string | null ) { this.setInputType( inputType ); }

  public get inputType(): string | null { return this.getInputType(); }

  /**
   * Get the input type. Input type is only relevant if this Node's primary sibling has tag name "INPUT".
   */
  public getInputType(): string | null {
    return this._inputType;
  }

  /**
   * By default the label will be prepended before the primary sibling in the PDOM. This
   * option allows you to instead have the label added after the primary sibling. Note: The label will always
   * be in front of the description sibling. If this flag is set with `appendDescription`, the order will be
   *
   * <container>
   *   <primary sibling/>
   *   <label sibling/>
   *   <description sibling/>
   * </container>
   */
  public setAppendLabel( appendLabel: boolean ): void {

    if ( this._appendLabel !== appendLabel ) {
      this._appendLabel = appendLabel;

      this.onPDOMContentChange();
    }
  }

  public set appendLabel( appendLabel: boolean ) { this.setAppendLabel( appendLabel ); }

  public get appendLabel(): boolean { return this.getAppendLabel(); }

  /**
   * Get whether the label sibling should be appended after the primary sibling.
   */
  public getAppendLabel(): boolean {
    return this._appendLabel;
  }

  /**
   * By default the label will be prepended before the primary sibling in the PDOM. This
   * option allows you to instead have the label added after the primary sibling. Note: The label will always
   * be in front of the description sibling. If this flag is set with `appendLabel`, the order will be
   *
   * <container>
   *   <primary sibling/>
   *   <label sibling/>
   *   <description sibling/>
   * </container>
   */
  public setAppendDescription( appendDescription: boolean ): void {

    if ( this._appendDescription !== appendDescription ) {
      this._appendDescription = appendDescription;

      this.onPDOMContentChange();
    }
  }

  public set appendDescription( appendDescription: boolean ) { this.setAppendDescription( appendDescription ); }

  public get appendDescription(): boolean { return this.getAppendDescription(); }

  /**
   * Get whether the description sibling should be appended after the primary sibling.
   */
  public getAppendDescription(): boolean {
    return this._appendDescription;
  }

  /**
   * Set the container parent tag name. By specifying this container parent, an element will be created that
   * acts as a container for this Node's primary sibling DOM Element and its label and description siblings.
   * This containerTagName will default to DEFAULT_LABEL_TAG_NAME, and be added to the PDOM automatically if
   * more than just the primary sibling is created.
   *
   * For instance, a button element with a label and description will be contained like the following
   * if the containerTagName is specified as 'section'.
   *
   * <section id='parent-container-trail-id'>
   *   <button>Press me!</button>
   *   <p>Button label</p>
   *   <p>Button description</p>
   * </section>
   */
  public setContainerTagName( tagName: string | null ): void {
    assert && assert( tagName === null || typeof tagName === 'string', `invalid tagName argument: ${tagName}` );

    if ( this._containerTagName !== tagName ) {
      this._containerTagName = tagName;
      this.onPDOMContentChange();
    }
  }

  public set containerTagName( tagName: string | null ) { this.setContainerTagName( tagName ); }

  public get containerTagName(): string | null { return this.getContainerTagName(); }

  /**
   * Get the tag name for the container parent element.
   */
  public getContainerTagName(): string | null {
    return this._containerTagName;
  }

  /**
   * Set the content of the label sibling for the this node.  The label sibling will default to the value of
   * DEFAULT_LABEL_TAG_NAME if no `labelTagName` is provided. If the label sibling is a `LABEL` html element,
   * then the `for` attribute will automatically be added, pointing to the Node's primary sibling.
   *
   * This method supports adding content in two ways, with HTMLElement.textContent and HTMLElement.innerHTML.
   * The DOM setter is chosen based on if the label passes the `containsFormattingTags`.
   *
   * Passing a null label value will not clear the whole label sibling, just the inner content of the DOM Element.
   */
  public setLabelContent( providedLabel: PDOMValueType | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    const label = unwrapProperty( providedLabel );

    if ( this._labelContent !== label ) {
      this._labelContent = label;

      // if trying to set labelContent, make sure that there is a labelTagName default
      if ( !this._labelTagName ) {
        this.setLabelTagName( DEFAULT_LABEL_TAG_NAME );
      }

      for ( let i = 0; i < this._pdomInstances.length; i++ ) {
        const peer = this._pdomInstances[ i ].peer!;
        peer.setLabelSiblingContent( this._labelContent );
      }
    }
  }

  public set labelContent( label: PDOMValueType | null ) { this.setLabelContent( label ); }

  public get labelContent(): string | null { return this.getLabelContent(); }

  /**
   * Get the content for this Node's label sibling DOM element.
   */
  public getLabelContent(): string | null {
    return this._labelContent;
  }

  /**
   * Set the inner content for the primary sibling of the PDOMPeers of this Node. Will be set as textContent
   * unless content is html which uses exclusively formatting tags. A node with inner content cannot
   * have accessible descendants because this content will override the HTML of descendants of this node.
   */
  public setInnerContent( providedContent: PDOMValueType | null ): void {
    this._innerContentProperty.setValueOrTargetProperty( this, null, providedContent );
  }

  public set innerContent( content: PDOMValueType | null ) { this.setInnerContent( content ); }

  public get innerContent(): string | null { return this.getInnerContent(); }

  /**
   * Get the inner content, the string that is the innerHTML or innerText for the Node's primary sibling.
   */
  public getInnerContent(): string | null {
    return this._innerContentProperty.value;
  }

  private onInnerContentPropertyChange( value: string | null ): void {
    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      const peer = this._pdomInstances[ i ].peer!;
      peer.setPrimarySiblingContent( value );
    }
  }

  /**
   * Set the description content for this Node's primary sibling. The description sibling tag name must support
   * innerHTML and textContent. If a description element does not exist yet, a default
   * DEFAULT_LABEL_TAG_NAME will be assigned to the descriptionTagName.
   */
  public setDescriptionContent( providedDescriptionContent: PDOMValueType | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    const descriptionContent = unwrapProperty( providedDescriptionContent );

    if ( this._descriptionContent !== descriptionContent ) {
      this._descriptionContent = descriptionContent;

      // if there is no description element, assume that a paragraph element should be used
      if ( !this._descriptionTagName ) {
        this.setDescriptionTagName( DEFAULT_DESCRIPTION_TAG_NAME );
      }

      for ( let i = 0; i < this._pdomInstances.length; i++ ) {
        const peer = this._pdomInstances[ i ].peer!;
        peer.setDescriptionSiblingContent( this._descriptionContent );
      }
    }
  }

  public set descriptionContent( textContent: PDOMValueType | null ) { this.setDescriptionContent( textContent ); }

  public get descriptionContent(): string | null { return this.getDescriptionContent(); }

  /**
   * Get the content for this Node's description sibling DOM Element.
   */
  public getDescriptionContent(): string | null {
    return this._descriptionContent;
  }

  /**
   * Set the ARIA role for this Node's primary sibling. According to the W3C, the ARIA role is read-only for a DOM
   * element.  So this will create a new DOM element for this Node with the desired role, and replace the old
   * element in the DOM. Note that the aria role can completely change the events that fire from an element,
   * especially when using a screen reader. For example, a role of `application` will largely bypass the default
   * behavior and logic of the screen reader, triggering keydown/keyup events even for buttons that would usually
   * only receive a "click" event.
   *
   * @param ariaRole - role for the element, see
   *                            https://www.w3.org/TR/html-aria/#allowed-aria-roles-states-and-properties
   *                            for a list of roles, states, and properties.
   */
  public setAriaRole( ariaRole: string | null ): void {
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
  }

  public set ariaRole( ariaRole: string | null ) { this.setAriaRole( ariaRole ); }

  public get ariaRole(): string | null { return this.getAriaRole(); }

  /**
   * Get the ARIA role representing this node.
   */
  public getAriaRole(): string | null {
    return this._ariaRole;
  }

  /**
   * Set the ARIA role for this node's container parent element.  According to the W3C, the ARIA role is read-only
   * for a DOM element. This will create a new DOM element for the container parent with the desired role, and
   * replace it in the DOM.
   *
   * @param ariaRole - role for the element, see
   *                            https://www.w3.org/TR/html-aria/#allowed-aria-roles-states-and-properties
   *                            for a list of roles, states, and properties.
   */
  public setContainerAriaRole( ariaRole: string | null ): void {
    assert && assert( ariaRole === null || typeof ariaRole === 'string' );

    if ( this._containerAriaRole !== ariaRole ) {

      this._containerAriaRole = ariaRole;

      // clear out the attribute
      if ( ariaRole === null ) {
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
  }

  public set containerAriaRole( ariaRole: string | null ) { this.setContainerAriaRole( ariaRole ); }

  public get containerAriaRole(): string | null { return this.getContainerAriaRole(); }

  /**
   * Get the ARIA role assigned to the container parent element.
   */
  public getContainerAriaRole(): string | null {
    return this._containerAriaRole;
  }

  /**
   * Set the aria-valuetext of this Node independently from the changing value, if necessary. Setting to null will
   * clear this attribute.
   */
  public setAriaValueText( providedAriaValueText: PDOMValueType | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    const ariaValueText = unwrapProperty( providedAriaValueText );

    if ( this._ariaValueText !== ariaValueText ) {
      this._ariaValueText = ariaValueText;

      if ( ariaValueText === null ) {
        this.removePDOMAttribute( 'aria-valuetext' );
      }
      else {
        this.setPDOMAttribute( 'aria-valuetext', ariaValueText );
      }
    }
  }

  public set ariaValueText( ariaValueText: PDOMValueType | null ) { this.setAriaValueText( ariaValueText ); }

  public get ariaValueText(): string | null { return this.getAriaValueText(); }

  /**
   * Get the value of the aria-valuetext attribute for this Node's primary sibling. If null, then the attribute
   * has not been set on the primary sibling.
   */
  public getAriaValueText(): string | null {
    return this._ariaValueText;
  }

  /**
   * Sets the namespace for the primary element (relevant for MathML/SVG/etc.)
   *
   * For example, to create a MathML element:
   * { tagName: 'math', pdomNamespace: 'http://www.w3.org/1998/Math/MathML' }
   *
   * or for SVG:
   * { tagName: 'svg', pdomNamespace: 'http://www.w3.org/2000/svg' }
   *
   * @param pdomNamespace - Null indicates no namespace.
   */
  public setPDOMNamespace( pdomNamespace: string | null ): this {
    assert && assert( pdomNamespace === null || typeof pdomNamespace === 'string' );

    if ( this._pdomNamespace !== pdomNamespace ) {
      this._pdomNamespace = pdomNamespace;

      // If the namespace changes, tear down the view and redraw the whole thing, there is no easy mutable solution here.
      this.onPDOMContentChange();
    }

    return this;
  }

  public set pdomNamespace( value: string | null ) { this.setPDOMNamespace( value ); }

  public get pdomNamespace(): string | null { return this.getPDOMNamespace(); }

  /**
   * Returns the accessible namespace (see setPDOMNamespace for more information).
   */
  public getPDOMNamespace(): string | null {
    return this._pdomNamespace;
  }

  /**
   * Sets the 'aria-label' attribute for labelling the Node's primary sibling. By using the
   * 'aria-label' attribute, the label will be read on focus, but can not be found with the
   * virtual cursor. This is one way to set a DOM Element's Accessible Name.
   *
   * @param providedAriaLabel - the text for the aria label attribute
   */
  public setAriaLabel( providedAriaLabel: PDOMValueType | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    const ariaLabel = unwrapProperty( providedAriaLabel );

    if ( this._ariaLabel !== ariaLabel ) {
      this._ariaLabel = ariaLabel;

      if ( ariaLabel === null ) {
        this.removePDOMAttribute( 'aria-label' );
      }
      else {
        this.setPDOMAttribute( 'aria-label', ariaLabel );
      }
    }
  }

  public set ariaLabel( ariaLabel: PDOMValueType | null ) { this.setAriaLabel( ariaLabel ); }

  public get ariaLabel(): string | null { return this.getAriaLabel(); }

  /**
   * Get the value of the aria-label attribute for this Node's primary sibling.
   */
  public getAriaLabel(): string | null {
    return this._ariaLabel;
  }

  /**
   * Set the focus highlight for this node. By default, the focus highlight will be a pink rectangle that
   * surrounds the node's local bounds.  If focus highlight is set to 'invisible', the node will not have
   * any highlighting when it receives focus.
   */
  public setFocusHighlight( focusHighlight: Highlight ): void {
    if ( this._focusHighlight !== focusHighlight ) {
      this._focusHighlight = focusHighlight;

      // if the focus highlight is layerable in the scene graph, update visibility so that it is only
      // visible when associated node has focus
      if ( this._focusHighlightLayerable ) {

        // if focus highlight is layerable, it must be a node in the scene graph
        assert && assert( focusHighlight instanceof Node ); // eslint-disable-line no-simple-type-checking-assertions

        // the highlight starts off invisible, HighlightOverlay will make it visible when this Node has DOM focus
        ( focusHighlight as Node ).visible = false;
      }

      this.focusHighlightChangedEmitter.emit();
    }
  }

  public set focusHighlight( focusHighlight: Highlight ) { this.setFocusHighlight( focusHighlight ); }

  public get focusHighlight(): Highlight { return this.getFocusHighlight(); }

  /**
   * Get the focus highlight for this node.
   */
  public getFocusHighlight(): Highlight {
    return this._focusHighlight;
  }

  /**
   * Setting a flag to break default and allow the focus highlight to be (z) layered into the scene graph.
   * This will set the visibility of the layered focus highlight, it will always be invisible until this node has
   * focus.
   */
  public setFocusHighlightLayerable( focusHighlightLayerable: boolean ): void {

    if ( this._focusHighlightLayerable !== focusHighlightLayerable ) {
      this._focusHighlightLayerable = focusHighlightLayerable;

      // if a focus highlight is defined (it must be a node), update its visibility so it is linked to focus
      // of the associated node
      if ( this._focusHighlight ) {
        assert && assert( this._focusHighlight instanceof Node );
        ( this._focusHighlight as Node ).visible = false;

        // emit that the highlight has changed and we may need to update its visual representation
        this.focusHighlightChangedEmitter.emit();
      }
    }
  }

  public set focusHighlightLayerable( focusHighlightLayerable: boolean ) { this.setFocusHighlightLayerable( focusHighlightLayerable ); }

  public get focusHighlightLayerable(): boolean { return this.getFocusHighlightLayerable(); }

  /**
   * Get the flag for if this node is layerable in the scene graph (or if it is always on top, like the default).
   */
  public getFocusHighlightLayerable(): boolean {
    return this._focusHighlightLayerable;
  }

  /**
   * Set whether or not this node has a group focus highlight. If this node has a group focus highlight, an extra
   * focus highlight will surround this node whenever a descendant node has focus. Generally
   * useful to indicate nested keyboard navigation. If true, the group focus highlight will surround
   * this node's local bounds. Otherwise, the Node will be used.
   *
   * TODO: Support more than one group focus highlight (multiple ancestors could have groupFocusHighlight), see https://github.com/phetsims/scenery/issues/708
   */
  public setGroupFocusHighlight( groupHighlight: Node | boolean ): void {
    this._groupFocusHighlight = groupHighlight;
  }

  public set groupFocusHighlight( groupHighlight: Node | boolean ) { this.setGroupFocusHighlight( groupHighlight ); }

  public get groupFocusHighlight(): Node | boolean { return this.getGroupFocusHighlight(); }

  /**
   * Get whether or not this node has a 'group' focus highlight, see setter for more information.
   */
  public getGroupFocusHighlight(): Node | boolean {
    return this._groupFocusHighlight;
  }

  /**
   * Very similar algorithm to setChildren in Node.js
   * @param ariaLabelledbyAssociations - list of associationObjects, see this._ariaLabelledbyAssociations.
   */
  public setAriaLabelledbyAssociations( ariaLabelledbyAssociations: Association[] ): void {
    let associationObject;
    let i;

    // validation if assert is enabled
    if ( assert ) {
      assert( Array.isArray( ariaLabelledbyAssociations ) );
      for ( i = 0; i < ariaLabelledbyAssociations.length; i++ ) {
        associationObject = ariaLabelledbyAssociations[ i ];
      }
    }

    // no work to be done if both are empty, return early
    if ( ariaLabelledbyAssociations.length === 0 && this._ariaLabelledbyAssociations.length === 0 ) {
      return;
    }

    const beforeOnly: Association[] = []; // Will hold all nodes that will be removed.
    const afterOnly: Association[] = []; // Will hold all nodes that will be "new" children (added)
    const inBoth: Association[] = []; // Child nodes that "stay". Will be ordered for the "after" case.

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
  }

  public set ariaLabelledbyAssociations( ariaLabelledbyAssociations: Association[] ) { this.setAriaLabelledbyAssociations( ariaLabelledbyAssociations ); }

  public get ariaLabelledbyAssociations(): Association[] { return this.getAriaLabelledbyAssociations(); }

  public getAriaLabelledbyAssociations(): Association[] {
    return this._ariaLabelledbyAssociations;
  }

  /**
   * Add an aria-labelledby association to this node. The data in the associationObject will be implemented like
   * "a peer's HTMLElement of this Node (specified with the string constant stored in `thisElementName`) will have an
   * aria-labelledby attribute with a value that includes the `otherNode`'s peer HTMLElement's id (specified with
   * `otherElementName`)."
   *
   * There can be more than one association because an aria-labelledby attribute's value can be a space separated
   * list of HTML ids, and not just a single id, see https://www.w3.org/WAI/GL/wiki/Using_aria-labelledby_to_concatenate_a_label_from_several_text_nodes
   */
  public addAriaLabelledbyAssociation( associationObject: Association ): void {

    // TODO: assert if this associationObject is already in the association objects list! https://github.com/phetsims/scenery/issues/832

    this._ariaLabelledbyAssociations.push( associationObject ); // Keep track of this association.

    // Flag that this node is is being labelled by the other node, so that if the other node changes it can tell
    // this node to restore the association appropriately.
    associationObject.otherNode._nodesThatAreAriaLabelledbyThisNode.push( this as unknown as Node );

    this.updateAriaLabelledbyAssociationsInPeers();
  }

  /**
   * Remove an aria-labelledby association object, see addAriaLabelledbyAssociation for more details
   */
  public removeAriaLabelledbyAssociation( associationObject: Association ): void {
    assert && assert( _.includes( this._ariaLabelledbyAssociations, associationObject ) );

    // remove the
    const removedObject = this._ariaLabelledbyAssociations.splice( _.indexOf( this._ariaLabelledbyAssociations, associationObject ), 1 );

    // remove the reference from the other node back to this node because we don't need it anymore
    removedObject[ 0 ].otherNode.removeNodeThatIsAriaLabelledByThisNode( this as unknown as Node );

    this.updateAriaLabelledbyAssociationsInPeers();
  }

  /**
   * Remove the reference to the node that is using this Node's ID as an aria-labelledby value (scenery-internal)
   */
  public removeNodeThatIsAriaLabelledByThisNode( node: Node ): void {
    const indexOfNode = _.indexOf( this._nodesThatAreAriaLabelledbyThisNode, node );
    assert && assert( indexOfNode >= 0 );
    this._nodesThatAreAriaLabelledbyThisNode.splice( indexOfNode, 1 );
  }

  /**
   * Trigger the view update for each PDOMPeer
   */
  public updateAriaLabelledbyAssociationsInPeers(): void {
    for ( let i = 0; i < this.pdomInstances.length; i++ ) {
      const peer = this.pdomInstances[ i ].peer!;
      peer.onAriaLabelledbyAssociationChange();
    }
  }

  /**
   * Update the associations for aria-labelledby (scenery-internal)
   */
  public updateOtherNodesAriaLabelledby(): void {

    // if any other nodes are aria-labelledby this Node, update those associations too. Since this node's
    // pdom content needs to be recreated, they need to update their aria-labelledby associations accordingly.
    for ( let i = 0; i < this._nodesThatAreAriaLabelledbyThisNode.length; i++ ) {
      const otherNode = this._nodesThatAreAriaLabelledbyThisNode[ i ];
      otherNode.updateAriaLabelledbyAssociationsInPeers();
    }
  }

  /**
   * The list of Nodes that are aria-labelledby this node (other node's peer element will have this Node's Peer element's
   * id in the aria-labelledby attribute
   */
  public getNodesThatAreAriaLabelledbyThisNode(): Node[] {
    return this._nodesThatAreAriaLabelledbyThisNode;
  }

  public get nodesThatAreAriaLabelledbyThisNode(): Node[] { return this.getNodesThatAreAriaLabelledbyThisNode(); }

  public setAriaDescribedbyAssociations( ariaDescribedbyAssociations: Association[] ): void {
    let associationObject;
    if ( assert ) {
      assert( Array.isArray( ariaDescribedbyAssociations ) );
      for ( let j = 0; j < ariaDescribedbyAssociations.length; j++ ) {
        associationObject = ariaDescribedbyAssociations[ j ];
      }
    }

    // no work to be done if both are empty
    if ( ariaDescribedbyAssociations.length === 0 && this._ariaDescribedbyAssociations.length === 0 ) {
      return;
    }

    const beforeOnly: Association[] = []; // Will hold all nodes that will be removed.
    const afterOnly: Association[] = []; // Will hold all nodes that will be "new" children (added)
    const inBoth: Association[] = []; // Child nodes that "stay". Will be ordered for the "after" case.
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
  }

  public set ariaDescribedbyAssociations( ariaDescribedbyAssociations: Association[] ) { this.setAriaDescribedbyAssociations( ariaDescribedbyAssociations ); }

  public get ariaDescribedbyAssociations(): Association[] { return this.getAriaDescribedbyAssociations(); }

  public getAriaDescribedbyAssociations(): Association[] {
    return this._ariaDescribedbyAssociations;
  }

  /**
   * Add an aria-describedby association to this node. The data in the associationObject will be implemented like
   * "a peer's HTMLElement of this Node (specified with the string constant stored in `thisElementName`) will have an
   * aria-describedby attribute with a value that includes the `otherNode`'s peer HTMLElement's id (specified with
   * `otherElementName`)."
   *
   * There can be more than one association because an aria-describedby attribute's value can be a space separated
   * list of HTML ids, and not just a single id, see https://www.w3.org/WAI/GL/wiki/Using_aria-labelledby_to_concatenate_a_label_from_several_text_nodes
   */
  public addAriaDescribedbyAssociation( associationObject: Association ): void {
    assert && assert( !_.includes( this._ariaDescribedbyAssociations, associationObject ), 'describedby association already registed' );

    this._ariaDescribedbyAssociations.push( associationObject ); // Keep track of this association.

    // Flag that this node is is being described by the other node, so that if the other node changes it can tell
    // this node to restore the association appropriately.
    associationObject.otherNode._nodesThatAreAriaDescribedbyThisNode.push( this as unknown as Node );

    // update the PDOMPeers with this aria-describedby association
    this.updateAriaDescribedbyAssociationsInPeers();
  }

  /**
   * Is this object already in the describedby association list
   */
  public hasAriaDescribedbyAssociation( associationObject: Association ): boolean {
    return _.includes( this._ariaDescribedbyAssociations, associationObject );
  }

  /**
   * Remove an aria-describedby association object, see addAriaDescribedbyAssociation for more details
   */
  public removeAriaDescribedbyAssociation( associationObject: Association ): void {
    assert && assert( _.includes( this._ariaDescribedbyAssociations, associationObject ) );

    // remove the
    const removedObject = this._ariaDescribedbyAssociations.splice( _.indexOf( this._ariaDescribedbyAssociations, associationObject ), 1 );

    // remove the reference from the other node back to this node because we don't need it anymore
    removedObject[ 0 ].otherNode.removeNodeThatIsAriaDescribedByThisNode( this as unknown as Node );

    this.updateAriaDescribedbyAssociationsInPeers();
  }

  /**
   * Remove the reference to the node that is using this Node's ID as an aria-describedby value (scenery-internal)
   */
  public removeNodeThatIsAriaDescribedByThisNode( node: Node ): void {
    const indexOfNode = _.indexOf( this._nodesThatAreAriaDescribedbyThisNode, node );
    assert && assert( indexOfNode >= 0 );
    this._nodesThatAreAriaDescribedbyThisNode.splice( indexOfNode, 1 );
  }

  /**
   * Trigger the view update for each PDOMPeer
   */
  public updateAriaDescribedbyAssociationsInPeers(): void {
    for ( let i = 0; i < this.pdomInstances.length; i++ ) {
      const peer = this.pdomInstances[ i ].peer!;
      peer.onAriaDescribedbyAssociationChange();
    }
  }

  /**
   * Update the associations for aria-describedby (scenery-internal)
   */
  public updateOtherNodesAriaDescribedby(): void {

    // if any other nodes are aria-describedby this Node, update those associations too. Since this node's
    // pdom content needs to be recreated, they need to update their aria-describedby associations accordingly.
    // TODO: only use unique elements of the array (_.unique)
    for ( let i = 0; i < this._nodesThatAreAriaDescribedbyThisNode.length; i++ ) {
      const otherNode = this._nodesThatAreAriaDescribedbyThisNode[ i ];
      otherNode.updateAriaDescribedbyAssociationsInPeers();
    }
  }

  /**
   * The list of Nodes that are aria-describedby this node (other node's peer element will have this Node's Peer element's
   * id in the aria-describedby attribute
   */
  public getNodesThatAreAriaDescribedbyThisNode(): Node[] {
    return this._nodesThatAreAriaDescribedbyThisNode;
  }

  public get nodesThatAreAriaDescribedbyThisNode(): Node[] { return this.getNodesThatAreAriaDescribedbyThisNode(); }

  public setActiveDescendantAssociations( activeDescendantAssociations: Association[] ): void {

    let associationObject;
    if ( assert ) {
      assert( Array.isArray( activeDescendantAssociations ) );
      for ( let j = 0; j < activeDescendantAssociations.length; j++ ) {
        associationObject = activeDescendantAssociations[ j ];
      }
    }

    // no work to be done if both are empty, safe to return early
    if ( activeDescendantAssociations.length === 0 && this._activeDescendantAssociations.length === 0 ) {
      return;
    }

    const beforeOnly: Association[] = []; // Will hold all nodes that will be removed.
    const afterOnly: Association[] = []; // Will hold all nodes that will be "new" children (added)
    const inBoth: Association[] = []; // Child nodes that "stay". Will be ordered for the "after" case.
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
  }

  public set activeDescendantAssociations( activeDescendantAssociations: Association[] ) { this.setActiveDescendantAssociations( activeDescendantAssociations ); }

  public get activeDescendantAssociations(): Association[] { return this.getActiveDescendantAssociations(); }

  public getActiveDescendantAssociations(): Association[] {
    return this._activeDescendantAssociations;
  }

  /**
   * Add an aria-activeDescendant association to this node. The data in the associationObject will be implemented like
   * "a peer's HTMLElement of this Node (specified with the string constant stored in `thisElementName`) will have an
   * aria-activeDescendant attribute with a value that includes the `otherNode`'s peer HTMLElement's id (specified with
   * `otherElementName`)."
   */
  public addActiveDescendantAssociation( associationObject: Association ): void {

    // TODO: assert if this associationObject is already in the association objects list! https://github.com/phetsims/scenery/issues/832
    this._activeDescendantAssociations.push( associationObject ); // Keep track of this association.

    // Flag that this node is is being described by the other node, so that if the other node changes it can tell
    // this node to restore the association appropriately.
    associationObject.otherNode._nodesThatAreActiveDescendantToThisNode.push( this as unknown as Node );

    // update the pdomPeers with this aria-activeDescendant association
    this.updateActiveDescendantAssociationsInPeers();
  }

  /**
   * Remove an aria-activeDescendant association object, see addActiveDescendantAssociation for more details
   */
  public removeActiveDescendantAssociation( associationObject: Association ): void {
    assert && assert( _.includes( this._activeDescendantAssociations, associationObject ) );

    // remove the
    const removedObject = this._activeDescendantAssociations.splice( _.indexOf( this._activeDescendantAssociations, associationObject ), 1 );

    // remove the reference from the other node back to this node because we don't need it anymore
    removedObject[ 0 ].otherNode.removeNodeThatIsActiveDescendantThisNode( this as unknown as Node );

    this.updateActiveDescendantAssociationsInPeers();
  }

  /**
   * Remove the reference to the node that is using this Node's ID as an aria-activeDescendant value (scenery-internal)
   */
  private removeNodeThatIsActiveDescendantThisNode( node: Node ): void {
    const indexOfNode = _.indexOf( this._nodesThatAreActiveDescendantToThisNode, node );
    assert && assert( indexOfNode >= 0 );
    this._nodesThatAreActiveDescendantToThisNode.splice( indexOfNode, 1 );

  }

  /**
   * Trigger the view update for each PDOMPeer
   */
  private updateActiveDescendantAssociationsInPeers(): void {
    for ( let i = 0; i < this.pdomInstances.length; i++ ) {
      const peer = this.pdomInstances[ i ].peer!;
      peer.onActiveDescendantAssociationChange();
    }
  }

  /**
   * Update the associations for aria-activeDescendant (scenery-internal)
   */
  public updateOtherNodesActiveDescendant(): void {

    // if any other nodes are aria-activeDescendant this Node, update those associations too. Since this node's
    // pdom content needs to be recreated, they need to update their aria-activeDescendant associations accordingly.
    // TODO: only use unique elements of the array (_.unique)
    for ( let i = 0; i < this._nodesThatAreActiveDescendantToThisNode.length; i++ ) {
      const otherNode = this._nodesThatAreActiveDescendantToThisNode[ i ];
      otherNode.updateActiveDescendantAssociationsInPeers();
    }
  }

  /**
   * The list of Nodes that are aria-activeDescendant this node (other node's peer element will have this Node's Peer element's
   * id in the aria-activeDescendant attribute
   */
  private getNodesThatAreActiveDescendantToThisNode(): Node[] {
    return this._nodesThatAreActiveDescendantToThisNode;
  }

  private get nodesThatAreActiveDescendantToThisNode() { return this.getNodesThatAreActiveDescendantToThisNode(); }


  /**
   * Sets the PDOM/DOM order for this Node. This includes not only focused items, but elements that can be
   * placed in the Parallel DOM. If provided, it will override the focus order between children (and
   * optionally arbitrary subtrees). If not provided, the focus order will default to the rendering order
   * (first children first, last children last), determined by the children array. A Node must be conected to a scene
   * graph (via children) in order for PDOM order to apply. Thus `setPDOMOrder` cannot be used in exchange for
   * setting a node as a child.
   *
   * In the general case, when an pdom order is specified, it's an array of nodes, with optionally one
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
   * - Nodes must be attached to a Display (in a scene graph) to be shown in an pdom order.
   * - You can't specify a node in more than one pdomOrder, and you can't specify duplicates of a value
   *   in an pdomOrder.
   * - You can't specify an ancestor of a node in that node's pdomOrder
   *   (e.g. this.pdomOrder = this.parents ).
   *
   * Note that specifying something in an pdomOrder will effectively remove it from all of its parents for
   * the pdom tree (so if you create `tmpNode.pdomOrder = [ a ]` then toss the tmpNode without
   * disposing it, `a` won't show up in the parallel DOM). If there is a need for that, disposing a Node
   * effectively removes its pdomOrder.
   *
   * See https://github.com/phetsims/scenery-phet/issues/365#issuecomment-381302583 for more information on the
   * decisions and design for this feature.
   */
  public setPDOMOrder( pdomOrder: ( Node | null )[] | null ): void {
    assert && assert( Array.isArray( pdomOrder ) || pdomOrder === null,
      `Array or null expected, received: ${pdomOrder}` );
    assert && pdomOrder && pdomOrder.forEach( ( node, index ) => {
      assert && assert( node === null || node instanceof Node,
        `Elements of pdomOrder should be either a Node or null. Element at index ${index} is: ${node}` );
    } );
    assert && pdomOrder && assert( ( this as unknown as Node ).getTrails( node => _.includes( pdomOrder, node ) ).length === 0, 'pdomOrder should not include any ancestors or the node itself' );

    // Only update if it has changed
    if ( this._pdomOrder !== pdomOrder ) {
      const oldPDOMOrder = this._pdomOrder;

      // Store our own reference to this, so client modifications to the input array won't silently break things.
      // See https://github.com/phetsims/scenery/issues/786
      this._pdomOrder = pdomOrder === null ? null : pdomOrder.slice();

      PDOMTree.pdomOrderChange( this as unknown as Node, oldPDOMOrder, pdomOrder );

      ( this as unknown as Node ).rendererSummaryRefreshEmitter.emit();
    }
  }

  public set pdomOrder( value: ( Node | null )[] | null ) { this.setPDOMOrder( value ); }

  public get pdomOrder(): ( Node | null )[] | null { return this.getPDOMOrder(); }

  /**
   * Returns the pdom (focus) order for this node.
   * If there is an existing array, this returns a copy of that array. This is important because clients may then
   * modify the array, and call setPDOMOrder - which is a no-op unless the array reference is different.
   */
  public getPDOMOrder(): ( Node | null )[] | null {
    if ( this._pdomOrder ) {
      return this._pdomOrder.slice( 0 ); // create a defensive copy
    }
    return this._pdomOrder;
  }

  /**
   * Returns whether this node has an pdomOrder that is effectively different than the default.
   *
   * NOTE: `null`, `[]` and `[null]` are all effectively the same thing, so this will return true for any of
   * those. Usage of `null` is recommended, as it doesn't create the extra object reference (but some code
   * that generates arrays may be more convenient).
   */
  public hasPDOMOrder(): boolean {
    return this._pdomOrder !== null &&
           this._pdomOrder.length !== 0 &&
           ( this._pdomOrder.length > 1 || this._pdomOrder[ 0 ] !== null );
  }

  /**
   * Returns our "PDOM parent" if available: the node that specifies this node in its pdomOrder.
   */
  public getPDOMParent(): Node | null {
    return this._pdomParent;
  }

  public get pdomParent(): Node | null { return this.getPDOMParent(); }

  /**
   * Returns the "effective" pdom children for the node (which may be different based on the order or other
   * excluded subtrees).
   *
   * If there is no pdomOrder specified, this is basically "all children that don't have pdom parents"
   * (a Node has a "PDOM parent" if it is specified in an pdomOrder).
   *
   * Otherwise (if it has an pdomOrder), it is the pdomOrder, with the above list of nodes placed
   * in at the location of the placeholder. If there is no placeholder, it acts like a placeholder was the last
   * element of the pdomOrder (see setPDOMOrder for more documentation information).
   *
   * NOTE: If you specify a child in the pdomOrder, it will NOT be double-included (since it will have an
   * PDOM parent).
   *
   * (scenery-internal)
   */
  public getEffectiveChildren(): Node[] {
    // Find all children without PDOM parents.
    const nonOrderedChildren = [];
    for ( let i = 0; i < ( this as unknown as Node )._children.length; i++ ) {
      const child = ( this as unknown as Node )._children[ i ];

      if ( !child._pdomParent ) {
        nonOrderedChildren.push( child );
      }
    }

    // Override the order, and replace the placeholder if it exists.
    if ( this.hasPDOMOrder() ) {
      const effectiveChildren = this.pdomOrder!.slice();

      const placeholderIndex = effectiveChildren.indexOf( null );

      // If we have a placeholder, replace its content with the children
      if ( placeholderIndex >= 0 ) {
        // for efficiency
        nonOrderedChildren.unshift( placeholderIndex, 1 );

        // @ts-expect-error - TODO: best way to type?
        Array.prototype.splice.apply( effectiveChildren, nonOrderedChildren );
      }
      // Otherwise, just add the normal things at the end
      else {
        Array.prototype.push.apply( effectiveChildren, nonOrderedChildren );
      }

      return effectiveChildren as Node[];
    }
    else {
      return nonOrderedChildren;
    }
  }

  /**
   * Hide completely from a screen reader and the browser by setting the hidden attribute on the node's
   * representative DOM element. If the sibling DOM Elements have a container parent, the container
   * should be hidden so that all PDOM elements are hidden as well.  Hiding the element will remove it from the focus
   * order.
   */
  public setPDOMVisible( visible: boolean ): void {
    if ( this._pdomVisible !== visible ) {
      this._pdomVisible = visible;

      this._pdomDisplaysInfo.onPDOMVisibilityChange( visible );
    }
  }

  public set pdomVisible( visible: boolean ) { this.setPDOMVisible( visible ); }

  public get pdomVisible(): boolean { return this.isPDOMVisible(); }

  /**
   * Get whether or not this node's representative DOM element is visible.
   */
  public isPDOMVisible(): boolean {
    return this._pdomVisible;
  }

  /**
   * Returns true if any of the PDOMInstances for the Node are globally visible and displayed in the PDOM. A
   * PDOMInstance is globally visible if Node and all ancestors are pdomVisible. PDOMInstance visibility is
   * updated synchronously, so this returns the most up-to-date information without requiring Display.updateDisplay
   */
  public isPDOMDisplayed(): boolean {
    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      if ( this._pdomInstances[ i ].isGloballyVisible() ) {
        return true;
      }
    }
    return false;
  }

  public get pdomDisplayed(): boolean { return this.isPDOMDisplayed(); }

  /**
   * Set the value of an input element.  Element must be a form element to support the value attribute. The input
   * value is converted to string since input values are generally string for HTML.
   */
  public setInputValue( value: PDOMValueType | number | null ): void {
    // If it's a Property, we'll just grab the initial value. See https://github.com/phetsims/scenery/issues/1442
    if ( value instanceof ReadOnlyProperty || value instanceof TinyProperty ) {
      value = value.value;
    }
    assert && assert( value === null || typeof value === 'string' || typeof value === 'number' );
    assert && this._tagName && assert( _.includes( FORM_ELEMENTS, this._tagName.toUpperCase() ), 'dom element must be a form element to support value' );

    // type cast
    value = `${value}`;

    if ( value !== this._inputValue ) {
      this._inputValue = value;

      for ( let i = 0; i < this.pdomInstances.length; i++ ) {
        const peer = this.pdomInstances[ i ].peer!;
        peer.onInputValueChange();
      }
    }
  }

  public set inputValue( value: PDOMValueType | number | null ) { this.setInputValue( value ); }

  public get inputValue(): string | number | null { return this.getInputValue(); }

  /**
   * Get the value of the element. Element must be a form element to support the value attribute.
   */
  public getInputValue(): string | number | null {
    return this._inputValue;
  }

  /**
   * Set whether or not the checked attribute appears on the dom elements associated with this Node's
   * pdom content.  This is only useful for inputs of type 'radio' and 'checkbox'. A 'checked' input
   * is considered selected to the browser and assistive technology.
   */
  public setPDOMChecked( checked: boolean ): void {

    if ( this._tagName ) {
      assert && assert( this._tagName.toUpperCase() === INPUT_TAG, 'Cannot set checked on a non input tag.' );
    }
    if ( this._inputType ) {
      assert && assert( INPUT_TYPES_THAT_SUPPORT_CHECKED.includes( this._inputType.toUpperCase() ), `inputType does not support checked: ${this._inputType}` );
    }

    if ( this._pdomChecked !== checked ) {
      this._pdomChecked = checked;

      this.setPDOMAttribute( 'checked', checked, {
        asProperty: true
      } );
    }
  }

  public set pdomChecked( checked: boolean ) { this.setPDOMChecked( checked ); }

  public get pdomChecked(): boolean { return this.getPDOMChecked(); }

  /**
   * Get whether or not the pdom input is 'checked'.
   */
  public getPDOMChecked(): boolean {
    return this._pdomChecked;
  }

  /**
   * Get an array containing all pdom attributes that have been added to this Node's primary sibling.
   */
  public getPDOMAttributes(): PDOMAttribute[] {
    return this._pdomAttributes.slice( 0 ); // defensive copy
  }

  public get pdomAttributes(): PDOMAttribute[] { return this.getPDOMAttributes(); }

  /**
   * Set a particular attribute or property for this Node's primary sibling, generally to provide extra semantic information for
   * a screen reader.
   *
   * @param attribute - string naming the attribute
   * @param value - the value for the attribute, if boolean, then it will be set as a javascript property on the HTMLElement rather than an attribute
   * @param [providedOptions]
   */
  public setPDOMAttribute( attribute: string, value: PDOMValueType | boolean | number, providedOptions?: SetPDOMAttributeOptions ): void {
    if ( !( typeof value === 'boolean' || typeof value === 'number' ) ) {
      value = unwrapProperty( value )!;
    }

    assert && providedOptions && assert( Object.getPrototypeOf( providedOptions ) === Object.prototype,
      'Extra prototype on pdomAttribute options object is a code smell' );
    assert && typeof value === 'string' && validate( value, Validation.STRING_WITHOUT_TEMPLATE_VARS_VALIDATOR );

    const options = optionize<SetPDOMAttributeOptions>()( {

      // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
      // for setting certain attributes (e.g. MathML).
      namespace: null,

      // set the "attribute" as a javascript property on the DOMElement instead
      asProperty: false,

      elementName: PDOMPeer.PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
    }, providedOptions );

    assert && assert( !ASSOCIATION_ATTRIBUTES.includes( attribute ), 'setPDOMAttribute does not support association attributes' );

    // if the pdom attribute already exists in the list, remove it - no need
    // to remove from the peers, existing attributes will simply be replaced in the DOM
    for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
      const currentAttribute = this._pdomAttributes[ i ];
      if ( currentAttribute.attribute === attribute &&
           currentAttribute.options.namespace === options.namespace &&
           currentAttribute.options.elementName === options.elementName ) {

        if ( currentAttribute.options.asProperty === options.asProperty ) {
          this._pdomAttributes.splice( i, 1 );
        }
        else {

          // Swapping asProperty setting strategies should remove the attribute so it can be set as a property.
          this.removePDOMAttribute( currentAttribute.attribute, currentAttribute.options );
        }
      }
    }

    this._pdomAttributes.push( {
      attribute: attribute,
      value: value,
      options: options
    } as PDOMAttribute );

    for ( let j = 0; j < this._pdomInstances.length; j++ ) {
      const peer = this._pdomInstances[ j ].peer!;
      peer.setAttributeToElement( attribute, value, options );
    }
  }

  /**
   * Remove a particular attribute, removing the associated semantic information from the DOM element.
   *
   * It is HIGHLY recommended that you never call this function from an attribute set with `asProperty:true`, see
   * setPDOMAttribute for the option details.
   *
   * @param attribute - name of the attribute to remove
   * @param [providedOptions]
   */
  public removePDOMAttribute( attribute: string, providedOptions?: RemovePDOMAttributeOptions ): void {
    assert && providedOptions && assert( Object.getPrototypeOf( providedOptions ) === Object.prototype,
      'Extra prototype on pdomAttribute options object is a code smell' );

    const options = optionize<RemovePDOMAttributeOptions>()( {

      // {string|null} - If non-null, will remove the attribute with the specified namespace. This can be required
      // for removing certain attributes (e.g. MathML).
      namespace: null,

      elementName: PDOMPeer.PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
    }, providedOptions );

    let attributeRemoved = false;
    for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
      if ( this._pdomAttributes[ i ].attribute === attribute &&
           this._pdomAttributes[ i ].options.namespace === options.namespace &&
           this._pdomAttributes[ i ].options.elementName === options.elementName ) {
        this._pdomAttributes.splice( i, 1 );
        attributeRemoved = true;
      }
    }
    assert && assert( attributeRemoved, `Node does not have pdom attribute ${attribute}` );

    for ( let j = 0; j < this._pdomInstances.length; j++ ) {
      const peer = this._pdomInstances[ j ].peer!;
      peer.removeAttributeFromElement( attribute, options );
    }
  }

  /**
   * Remove all attributes from this node's dom element.
   */
  public removePDOMAttributes(): void {

    // all attributes currently on this Node's primary sibling
    const attributes = this.getPDOMAttributes();

    for ( let i = 0; i < attributes.length; i++ ) {
      const attribute = attributes[ i ].attribute;
      this.removePDOMAttribute( attribute );
    }
  }

  /**
   * Remove a particular attribute, removing the associated semantic information from the DOM element.
   *
   * @param attribute - name of the attribute to remove
   * @param [providedOptions]
   */
  public hasPDOMAttribute( attribute: string, providedOptions?: HasPDOMAttributeOptions ): boolean {
    assert && providedOptions && assert( Object.getPrototypeOf( providedOptions ) === Object.prototype,
      'Extra prototype on pdomAttribute options object is a code smell' );

    const options = optionize<HasPDOMAttributeOptions>()( {

      // {string|null} - If non-null, will remove the attribute with the specified namespace. This can be required
      // for removing certain attributes (e.g. MathML).
      namespace: null,

      elementName: PDOMPeer.PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
    }, providedOptions );

    let attributeFound = false;
    for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
      if ( this._pdomAttributes[ i ].attribute === attribute &&
           this._pdomAttributes[ i ].options.namespace === options.namespace &&
           this._pdomAttributes[ i ].options.elementName === options.elementName ) {
        attributeFound = true;
      }
    }
    return attributeFound;
  }

  /**
   * Add the class to the PDOM element's classList. The PDOM is generally invisible,
   * but some styling occasionally has an impact on semantics so it is necessary to set styles.
   * Add a class with this function and define the style in stylesheets (likely SceneryStyle).
   */
  public setPDOMClass( className: string, providedOptions?: SetPDOMClassOptions ): void {

    const options = optionize<SetPDOMClassOptions>()( {
      elementName: PDOMPeer.PRIMARY_SIBLING
    }, providedOptions );

    // if we already have the provided className set to the sibling, do nothing
    for ( let i = 0; i < this._pdomClasses.length; i++ ) {
      const currentClass = this._pdomClasses[ i ];
      if ( currentClass.className === className && currentClass.options.elementName === options.elementName ) {
        return;
      }
    }

    this._pdomClasses.push( { className: className, options: options } );

    for ( let j = 0; j < this._pdomInstances.length; j++ ) {
      const peer = this._pdomInstances[ j ].peer!;
      peer.setClassToElement( className, options );
    }
  }

  /**
   * Remove a class from the classList of one of the elements for this Node.
   */
  public removePDOMClass( className: string, providedOptions?: RemovePDOMClassOptions ): void {

    const options = optionize<RemovePDOMClassOptions>()( {
      elementName: PDOMPeer.PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
    }, providedOptions );

    let classRemoved = false;
    for ( let i = 0; i < this._pdomClasses.length; i++ ) {
      if ( this._pdomClasses[ i ].className === className &&
           this._pdomClasses[ i ].options.elementName === options.elementName ) {
        this._pdomClasses.splice( i, 1 );
        classRemoved = true;
      }
    }
    assert && assert( classRemoved, `Node does not have pdom attribute ${className}` );

    for ( let j = 0; j < this._pdomClasses.length; j++ ) {
      const peer = this.pdomInstances[ j ].peer!;
      peer.removeClassFromElement( className, options );
    }
  }

  /**
   * Get the list of classes assigned to PDOM elements for this Node.
   */
  public getPDOMClasses(): PDOMClass[] {
    return this._pdomClasses.slice( 0 ); // defensive copy
  }

  public get pdomClasses(): PDOMClass[] { return this.getPDOMClasses(); }

  /**
   * Make the DOM element explicitly focusable with a tab index. Native HTML form elements will generally be in
   * the navigation order without explicitly setting focusable.  If these need to be removed from the navigation
   * order, call setFocusable( false ).  Removing an element from the focus order does not hide the element from
   * assistive technology.
   *
   * @param focusable - null to use the default browser focus for the primary element
   */
  public setFocusable( focusable: boolean | null ): void {
    assert && assert( focusable === null || typeof focusable === 'boolean' );

    if ( this._focusableOverride !== focusable ) {
      this._focusableOverride = focusable;

      for ( let i = 0; i < this._pdomInstances.length; i++ ) {

        // after the override is set, update the focusability of the peer based on this node's value for focusable
        // which may be true or false (but not null)
        // assert && assert( typeof this.focusable === 'boolean' );
        assert && assert( this._pdomInstances[ i ].peer, 'Peer required to set focusable.' );
        this._pdomInstances[ i ].peer!.setFocusable( this.focusable );
      }
    }
  }

  public set focusable( isFocusable: boolean | null ) { this.setFocusable( isFocusable ); }

  public get focusable(): boolean { return this.isFocusable(); }

  /**
   * Get whether or not the node is focusable. Use the focusOverride, and then default to browser defined
   * focusable elements.
   */
  public isFocusable(): boolean {
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
  }

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
   */
  public setPDOMTransformSourceNode( node: Node | null ): void {
    this._pdomTransformSourceNode = node;

    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      this._pdomInstances[ i ].peer!.setPDOMTransformSourceNode( this._pdomTransformSourceNode );
    }
  }

  public set pdomTransformSourceNode( node: Node | null ) { this.setPDOMTransformSourceNode( node ); }

  public get pdomTransformSourceNode(): Node | null { return this.getPDOMTransformSourceNode(); }

  /**
   * Get the source Node that controls positioning of the primary sibling in the global coordinate frame. See
   * setPDOMTransformSourceNode for more in depth information.
   */
  public getPDOMTransformSourceNode(): Node | null {
    return this._pdomTransformSourceNode;
  }

  /**
   * Sets whether the PDOM sibling elements are positioned in the correct place in the viewport. Doing so is a
   * requirement for custom gestures on touch based screen readers. However, doing this DOM layout is expensive so
   * only do this when necessary. Generally only needed for elements that utilize a "double tap and hold" gesture
   * to drag and drop.
   *
   * Positioning the PDOM element will caused some screen readers to send both click and pointer events to the
   * location of the Node in global coordinates. Do not position elements that use click listeners since activation
   * will fire twice (once for the pointer event listeners and once for the click event listeners).
   */
  public setPositionInPDOM( positionInPDOM: boolean ): void {
    this._positionInPDOM = positionInPDOM;

    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      this._pdomInstances[ i ].peer!.setPositionInPDOM( positionInPDOM );
    }
  }

  public set positionInPDOM( positionInPDOM: boolean ) { this.setPositionInPDOM( positionInPDOM ); }

  public get positionInPDOM(): boolean { return this.getPositionInPDOM(); }

  /**
   * Gets whether or not we are positioning the PDOM sibling elements. See setPositionInPDOM().
   */
  public getPositionInPDOM(): boolean {
    return this._positionInPDOM;
  }

  /**
   * This function should be used sparingly as a workaround. If used, any DOM input events received from the label
   * sibling will not be dispatched as SceneryEvents in Input.js. The label sibling may receive input by screen
   * readers if the virtual cursor is over it. That is usually fine, but there is a bug with NVDA and Firefox where
   * both the label sibling AND primary sibling receive events in this case, and both bubble up to the root of the
   * PDOM, and so we would otherwise dispatch two SceneryEvents instead of one.
   *
   * See https://github.com/phetsims/a11y-research/issues/156 for more information.
   */
  public setExcludeLabelSiblingFromInput(): void {
    this.excludeLabelSiblingFromInput = true;
    this.onPDOMContentChange();
  }

  /**
   * Return true if this Node is a PhET-iO archetype or it is a Node descendant of a PhET-iO archetype.
   * See https://github.com/phetsims/joist/issues/817
   */
  public isInsidePhetioArchetype( node: Node = ( this as unknown as Node ) ): boolean {
    if ( node.isPhetioInstrumented() ) {
      return node.phetioIsArchetype;
    }
    for ( let i = 0; i < node.parents.length; i++ ) {
      if ( this.isInsidePhetioArchetype( node.parents[ i ] ) ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Alert on all interactive description utteranceQueues located on each connected Display. See
   * Node.getConnectedDisplays. Note that if your Node is not connected to a Display, this function will have
   * no effect.
   */
  public alertDescriptionUtterance( utterance: TAlertable ): void {

    // No description should be alerted if setting PhET-iO state, see https://github.com/phetsims/scenery/issues/1397
    if ( _.hasIn( window, 'phet.phetio.phetioEngine.phetioStateEngine' ) &&
         phet.phetio.phetioEngine.phetioStateEngine.isSettingStateProperty.value ) {
      return;
    }

    // No description should be alerted if an archetype of a PhET-iO dynamic element, see https://github.com/phetsims/joist/issues/817
    if ( Tandem.PHET_IO_ENABLED && this.isInsidePhetioArchetype() ) {
      return;
    }

    const connectedDisplays = ( this as unknown as Node ).getConnectedDisplays();
    for ( let i = 0; i < connectedDisplays.length; i++ ) {
      const display = connectedDisplays[ i ];
      if ( display.isAccessible() ) {

        // Don't use `forEachUtterance` to prevent creating a closure for each usage of this function
        display.descriptionUtteranceQueue.addToBack( utterance );
      }
    }
  }

  /**
   * Apply a callback on each utteranceQueue that this Node has a connection to (via Display). Note that only
   * accessible Displays have utteranceQueues that this function will interface with.
   */
  public forEachUtteranceQueue( callback: ( queue: UtteranceQueue ) => void ): void {
    const connectedDisplays = ( this as unknown as Node ).getConnectedDisplays();

    // If you run into this assertion, talk to @jessegreenberg and @zepumph, because it is quite possible we would
    // remove this assertion for your case.
    assert && assert( connectedDisplays.length > 0,
      'must be connected to a display to use UtteranceQueue features' );

    for ( let i = 0; i < connectedDisplays.length; i++ ) {
      const display = connectedDisplays[ i ];
      if ( display.isAccessible() ) {
        callback( display.descriptionUtteranceQueue );
      }
    }
  }

  /***********************************************************************************************************/
  // SCENERY-INTERNAL AND PRIVATE METHODS
  /***********************************************************************************************************/

  /**
   * Used to get a list of all settable options and their current values. (scenery-internal)
   *
   * @returns - keys are all accessibility option keys, and the values are the values of those properties
   * on this node.
   */
  public getBaseOptions(): ParallelDOMOptions {

    const currentOptions: ParallelDOMOptions = {};

    for ( let i = 0; i < ACCESSIBILITY_OPTION_KEYS.length; i++ ) {
      const optionName = ACCESSIBILITY_OPTION_KEYS[ i ];

      // @ts-expect-error - Not sure of a great way to do this
      currentOptions[ optionName ] = this[ optionName ];
    }

    return currentOptions;
  }

  /**
   * Returns a recursive data structure that represents the nested ordering of pdom content for this Node's
   * subtree. Each "Item" will have the type { trail: {Trail}, children: {Array.<Item>} }, forming a tree-like
   * structure. (scenery-internal)
   */
  public getNestedPDOMOrder(): { trail: Trail; children: Node[] }[] {
    const currentTrail = new Trail( this as unknown as Node );
    let pruneStack: Node[] = []; // A list of nodes to prune

    // {Array.<Item>} - The main result we will be returning. It is the top-level array where child items will be
    // inserted.
    const result: { trail: Trail; children: Node[] }[] = [];

    // {Array.<Array.<Item>>} A stack of children arrays, where we should be inserting items into the top array.
    // We will start out with the result, and as nested levels are added, the children arrays of those items will be
    // pushed and poppped, so that the top array on this stack is where we should insert our next child item.
    const nestedChildStack = [ result ];

    function addTrailsForNode( node: Node, overridePruning: boolean ): void {
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
      // of creating duplicate references in the traversal order).
      if ( pruneCount > 1 || ( pruneCount === 1 && !overridePruning ) ) {
        return;
      }

      // Pushing item and its children array, if has pdom content
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
      pruneStack = pruneStack.concat( arrayPDOMOrder as Node[] );

      // Visiting trails to ordered nodes.
      // @ts-expect-error
      _.each( arrayPDOMOrder, ( descendant: Node ) => {
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

      // Popping children array if has pdom content
      if ( node.hasPDOMContent ) {
        nestedChildStack.pop();
      }
    }

    addTrailsForNode( ( this as unknown as Node ), false );

    return result;
  }

  /**
   * Sets the pdom content for a Node. See constructor for more information. Not part of the ParallelDOM
   * API (scenery-internal)
   */
  private onPDOMContentChange(): void {

    PDOMTree.pdomContentChange( this as unknown as Node );

    // recompute the heading level for this node if it is using the pdomHeading API.
    this._pdomHeading && this.computeHeadingLevel();

    ( this as unknown as Node ).rendererSummaryRefreshEmitter.emit();
  }

  /**
   * Returns whether or not this Node has any representation for the Parallel DOM.
   * Note this is still true if the content is pdomVisible=false or is otherwise hidden.
   */
  public get hasPDOMContent(): boolean {
    return !!this._tagName;
  }

  /**
   * Called when the node is added as a child to this node AND the node's subtree contains pdom content.
   * We need to notify all Displays that can see this change, so that they can update the PDOMInstance tree.
   */
  protected onPDOMAddChild( node: Node ): void {
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( `onPDOMAddChild n#${node.id} (parent:n#${( this as unknown as Node ).id})` );
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

    // Find descendants with pdomOrders and check them against all of their ancestors/self
    assert && ( function recur( descendant ) {
      // Prune the search (because milliseconds don't grow on trees, even if we do have assertions enabled)
      if ( descendant._rendererSummary.hasNoPDOM() ) { return; }

      descendant.pdomOrder && assert( descendant.getTrails( node => _.includes( descendant.pdomOrder, node ) ).length === 0, 'pdomOrder should not include any ancestors or the node itself' );
    } )( node );

    assert && PDOMTree.auditNodeForPDOMCycles( this as unknown as Node );

    this._pdomDisplaysInfo.onAddChild( node );

    PDOMTree.addChild( this as unknown as Node, node );

    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
  }

  /**
   * Called when the node is removed as a child from this node AND the node's subtree contains pdom content.
   * We need to notify all Displays that can see this change, so that they can update the PDOMInstance tree.
   */
  protected onPDOMRemoveChild( node: Node ): void {
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( `onPDOMRemoveChild n#${node.id} (parent:n#${( this as unknown as Node ).id})` );
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

    this._pdomDisplaysInfo.onRemoveChild( node );

    PDOMTree.removeChild( this as unknown as Node, node );

    // make sure that the associations for aria-labelledby and aria-describedby are updated for nodes associated
    // to this Node (they are pointing to this Node's IDs). https://github.com/phetsims/scenery/issues/816
    node.updateOtherNodesAriaLabelledby();
    node.updateOtherNodesAriaDescribedby();
    node.updateOtherNodesActiveDescendant();

    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
  }

  /**
   * Called when this node's children are reordered (with nothing added/removed).
   */
  protected onPDOMReorderedChildren(): void {
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( `onPDOMReorderedChildren (parent:n#${( this as unknown as Node ).id})` );
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

    PDOMTree.childrenOrderChange( this as unknown as Node );

    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
  }

  /**
   * Handles linking and checking child PhET-iO Properties such as Node.visibleProperty and Node.enabledProperty.
   */
  public updateLinkedElementForProperty<T>( tandemName: string, oldProperty?: TProperty<T> | null, newProperty?: TProperty<T> | null ): void {
    assert && assert( oldProperty !== newProperty, 'should not be called on same values' );

    // Only update linked elements if this Node is instrumented for PhET-iO
    if ( this.isPhetioInstrumented() ) {

      oldProperty && oldProperty instanceof ReadOnlyProperty && oldProperty.isPhetioInstrumented() && oldProperty instanceof PhetioObject && this.removeLinkedElements( oldProperty );

      const tandem = this.tandem.createTandem( tandemName );
      if ( newProperty && newProperty instanceof ReadOnlyProperty && newProperty.isPhetioInstrumented() && newProperty instanceof PhetioObject && tandem !== newProperty.tandem ) {
        this.addLinkedElement( newProperty, { tandem: tandem } );
      }
    }
  }

  /*---------------------------------------------------------------------------*/
  //
  // PDOM Instance handling

  /**
   * Returns a reference to the pdom instances array. (scenery-internal)
   */
  public getPDOMInstances(): PDOMInstance[] {
    return this._pdomInstances;
  }

  public get pdomInstances(): PDOMInstance[] { return this.getPDOMInstances(); }

  /**
   * Adds an PDOMInstance reference to our array. (scenery-internal)
   */
  public addPDOMInstance( pdomInstance: PDOMInstance ): void {
    this._pdomInstances.push( pdomInstance );
  }

  /**
   * Removes an PDOMInstance reference from our array. (scenery-internal)
   */
  public removePDOMInstance( pdomInstance: PDOMInstance ): void {
    const index = _.indexOf( this._pdomInstances, pdomInstance );
    assert && assert( index !== -1, 'Cannot remove an PDOMInstance from a Node if it was not there' );
    this._pdomInstances.splice( index, 1 );
  }

  public static BASIC_ACCESSIBLE_NAME_BEHAVIOR( node: Node, options: ParallelDOMOptions, accessibleName: PDOMValueType ): ParallelDOMOptions {
    if ( node.tagName === 'input' ) {
      options.labelTagName = 'label';
      options.labelContent = accessibleName;
    }
    else if ( PDOMUtils.tagNameSupportsContent( node.tagName! ) ) {
      options.innerContent = accessibleName;
    }
    else {
      options.ariaLabel = accessibleName;
    }
    return options;
  }

  public static HELP_TEXT_BEFORE_CONTENT( node: Node, options: ParallelDOMOptions, helpText: PDOMValueType ): ParallelDOMOptions {
    options.descriptionTagName = PDOMUtils.DEFAULT_DESCRIPTION_TAG_NAME;
    options.descriptionContent = helpText;
    options.appendDescription = false;
    return options;
  }

  public static HELP_TEXT_AFTER_CONTENT( node: Node, options: ParallelDOMOptions, helpText: PDOMValueType ): ParallelDOMOptions {
    options.descriptionTagName = PDOMUtils.DEFAULT_DESCRIPTION_TAG_NAME;
    options.descriptionContent = helpText;
    options.appendDescription = true;
    return options;
  }
}

scenery.register( 'ParallelDOM', ParallelDOM );
export { ACCESSIBILITY_OPTION_KEYS };
