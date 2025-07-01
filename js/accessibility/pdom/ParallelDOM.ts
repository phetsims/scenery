// Copyright 2021-2025, University of Colorado Boulder

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
 * #Input listeners
 * ParallelDOM is the primary way we listen to keyboard events in scenery. See TInputListener for supported keyboard
 * events that you can add. Note that the input events from the DOM that your ParallelDOM instance will receive is
 * dependent on what the DOM Element is (see tagName).
 *
 * NOTE: Be VERY careful about mutating ParallelDOM content in input listeners, this can result in events being dropped.
 * For example, if you press enter on a 'button', you would expect a keydown event followed by a click event, but if the
 * keydown listener changes the tagName to 'div', the click event will not occur.
 * --------------------------------------------------------------------------------------------------------------------
 *
 * For additional accessibility options, please see the options listed in ACCESSIBILITY_OPTION_KEYS. To understand the
 * PDOM more, see PDOMPeer, which manages the DOM Elements for a Node. For more documentation on Scenery, Nodes,
 * and the scene graph, please see http://phetsims.github.io/scenery/
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import ReadOnlyProperty from '../../../../axon/js/ReadOnlyProperty.js';
import TEmitter from '../../../../axon/js/TEmitter.js';
import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import TinyForwardingProperty from '../../../../axon/js/TinyForwardingProperty.js';
import TProperty from '../../../../axon/js/TProperty.js';
import TReadOnlyProperty, { isTReadOnlyProperty } from '../../../../axon/js/TReadOnlyProperty.js';
import validate from '../../../../axon/js/validate.js';
import Validation from '../../../../axon/js/Validation.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import Shape from '../../../../kite/js/Shape.js';
import arrayDifference from '../../../../phet-core/js/arrayDifference.js';
import arrayRemove from '../../../../phet-core/js/arrayRemove.js';
import optionize from '../../../../phet-core/js/optionize.js';
import PickOptional from '../../../../phet-core/js/types/PickOptional.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import isSettingPhetioStateProperty from '../../../../tandem/js/isSettingPhetioStateProperty.js';
import PhetioObject, { PhetioObjectOptions } from '../../../../tandem/js/PhetioObject.js';
import Tandem from '../../../../tandem/js/Tandem.js';
import { AlertableNoUtterance, TAlertable } from '../../../../utterance-queue/js/Utterance.js';
import type UtteranceQueue from '../../../../utterance-queue/js/UtteranceQueue.js';
import type Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import Trail from '../../util/Trail.js';

import { Highlight } from '../Highlight.js';
import PDOMDisplaysInfo from './PDOMDisplaysInfo.js';
import type PDOMInstance from './PDOMInstance.js';
import PDOMTree from './PDOMTree.js';
import PDOMUtils from './PDOMUtils.js';
import { PEER_CONTAINER_PARENT } from './PEER_CONTAINER_PARENT.js';
import { PEER_PRIMARY_SIBLING } from './PEER_PRIMARY_SIBLING.js';

const INPUT_TAG = PDOMUtils.TAGS.INPUT;
const P_TAG = PDOMUtils.TAGS.P;
const DIV_TAG = PDOMUtils.TAGS.DIV;

// default tag names for siblings
const DEFAULT_TAG_NAME = DIV_TAG;
const DEFAULT_DESCRIPTION_TAG_NAME = P_TAG;
const DEFAULT_LABEL_TAG_NAME = P_TAG;

export type PDOMValueType = string | TReadOnlyProperty<string> | null;
export type LimitPanDirection = 'horizontal' | 'vertical';

const unwrapProperty = ( valueOrProperty: PDOMValueType ): string | null => {
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
  // currently focused Node. We want the focusability to update correctly.
  'focusable',
  'tagName',

  /*
   * Higher Level API Functions
   */
  'accessibleName',
  'accessibleNameBehavior',
  'accessibleHelpText',
  'accessibleHelpTextBehavior',
  'accessibleParagraph',
  'accessibleParagraphBehavior',

  /*
   * Lower Level API Functions
   */
  'accessibleHeading',
  'accessibleHeadingIncrement',
  'accessibleRoleDescription',

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

  'accessibleParagraphContent',

  'focusHighlight',
  'focusHighlightLayerable',
  'groupFocusHighlight',
  'pdomVisibleProperty',
  'pdomVisible',
  'pdomOrder',

  'pdomAttributes',

  'ariaLabelledbyAssociations',
  'ariaDescribedbyAssociations',
  'activeDescendantAssociations',

  'focusPanTargetBoundsProperty',
  'limitPanDirection',

  'positionInPDOM',

  'pdomTransformSourceNode'
];

type ParallelDOMSelfOptions = {
  focusable?: boolean | null; // Sets whether the Node can receive keyboard focus
  tagName?: string | null; // Sets the tag name for the primary sibling DOM element in the parallel DOM, should be first

  /*
   * Higher Level API Functions
   */
  accessibleName?: PDOMValueType; // Sets the accessible name for this Node, see setAccessibleName() for more information.
  accessibleParagraph?: PDOMValueType; // Sets the accessible paragraph for this Node, see setAccessibleParagraph() for more information.
  accessibleHelpText?: PDOMValueType; // Sets the help text for this Node, see setAccessibleHelpText() for more information

  /*
   * Lower Level API Functions
   */
  accessibleNameBehavior?: AccessibleNameBehaviorFunction; // Sets the implementation for the accessibleName, see setAccessibleNameBehavior() for more information
  accessibleHelpTextBehavior?: AccessibleHelpTextBehaviorFunction; // Sets the implementation for the accessibleHelpText, see setAccessibleHelpTextBehavior() for more// information
  accessibleParagraphBehavior?: AccessibleParagraphBehaviorFunction; // Sets the implementation for the accessibleParagraph, see setAccessibleParagraphBehavior() for more information

  accessibleHeading?: PDOMValueType; // Sets the heading text for this Node, see setAccessibleHeading() for more information
  accessibleHeadingIncrement?: number; // Sets the heading level increment for this Node, see setAccessibleHeadingIncrement() for more information

  containerTagName?: string | null; // Sets the tag name for an [optional] element that contains this Node's siblings
  containerAriaRole?: string | null; // Sets the ARIA role for the container parent DOM element

  innerContent?: PDOMValueType; // Sets the inner text or HTML for a Node's primary sibling element
  inputType?: string | null; // Sets the input type for the primary sibling DOM element, only relevant if tagName is 'input'
  inputValue?: PDOMValueType | number; // Sets the input value for the primary sibling DOM element, only relevant if tagName is 'input'
  pdomChecked?: boolean; // Sets the 'checked' state for inputs of type 'radio' and 'checkbox'
  pdomNamespace?: string | null; // Sets the namespace for the primary element
  ariaLabel?: PDOMValueType; // Sets the value of the 'aria-label' attribute on the primary sibling of this Node
  ariaRole?: string | null; // Sets the ARIA role for the primary sibling of this Node
  ariaValueText?: PDOMValueType; // sets the aria-valuetext attribute of the primary sibling
  accessibleRoleDescription?: PDOMValueType; // Sets the aria-roledescription for the primary sibling

  labelTagName?: string | null; // Sets the tag name for the DOM element sibling labeling this Node
  labelContent?: PDOMValueType; // Sets the label content for the Node
  appendLabel?: boolean; // Sets the label sibling to come after the primary sibling in the PDOM

  descriptionTagName?: string | null; // Sets the tag name for the DOM element sibling describing this Node
  descriptionContent?: PDOMValueType; // Sets the description content for the Node
  appendDescription?: boolean; // Sets the description sibling to come after the primary sibling in the PDOM

  accessibleParagraphContent?: PDOMValueType; // Sets the accessible paragraph content for the Node

  focusHighlight?: Highlight; // Sets the focus highlight for the Node
  focusHighlightLayerable?: boolean; //lag to determine if the focus highlight Node can be layered in the scene graph
  groupFocusHighlight?: Node | boolean; // Sets the outer focus highlight for this Node when a descendant has focus
  pdomVisibleProperty?: TReadOnlyProperty<boolean> | null;
  pdomVisible?: boolean; // Sets whether or not the Node's DOM element is visible in the parallel DOM
  pdomOrder?: ( Node | null )[] | null; // Modifies the order of accessible navigation

  pdomAttributes?: PDOMAttribute[]; // Sets a list of attributes all at once, see setPDOMAttributes().

  ariaLabelledbyAssociations?: Association[]; // sets the list of aria-labelledby associations between from this Node to others (including itself)
  ariaDescribedbyAssociations?: Association[]; // sets the list of aria-describedby associations between from this Node to others (including itself)
  activeDescendantAssociations?: Association[]; // sets the list of aria-activedescendant associations between from this Node to others (including itself)

  focusPanTargetBoundsProperty?: TReadOnlyProperty<Bounds2> | null; // A Property with bounds that describe the bounds of this Node that should remain displayed by the global AnimatedPanZoomListener
  limitPanDirection?: LimitPanDirection | null; // A constraint on the direction of panning when interacting with this Node.

  positionInPDOM?: boolean; // Sets whether the Node's DOM elements are positioned in the viewport

  pdomTransformSourceNode?: Node | null; // { sets the Node that controls primary sibling element positioning in the display, see setPDOMTransformSourceNode()
};

// Most options use null for their default behavior, see the setters for each option for a description of how null
// behaves as a default.
export type ParallelDOMOptions = ParallelDOMSelfOptions & PhetioObjectOptions;

// Removes all options from T that are in ParallelDOMSelfOptions.
export type RemoveParallelDOMOptions<T extends ParallelDOMOptions> = StrictOmit<T, keyof ParallelDOMSelfOptions>;

// Removes all options from T that are in ParallelDOMSelfOptions, except for the most fundamental ones.
// This is useful for creating a ParallelDOM subclass that only exposes these high-level options while implementing
// accessibility with the lower-level API.
export type TrimParallelDOMOptions<T extends ParallelDOMSelfOptions> = RemoveParallelDOMOptions<T> &
  PickOptional<ParallelDOMSelfOptions, 'accessibleName' | 'accessibleHelpText' | 'accessibleParagraph' | 'accessibleHeading' | 'accessibleHeadingIncrement' | 'focusable' | 'pdomVisible'>;

type PDOMAttribute = {
  attribute: string;
  value: Exclude<PDOMValueType, null> | boolean | number;
  listener?: ( ( rawValue: string | boolean | number ) => void ) | null;
  options?: SetPDOMAttributeOptions;
};

type PDOMClass = {
  className: string;
  options: SetPDOMClassOptions;
};

export type Association = {
  otherNode: Node;
  otherElementName: string;
  thisElementName: string;
};

type SetPDOMAttributeOptions = {
  namespace?: string | null;
  type?: 'attribute' | 'property'; // javascript Property instead of AXON/Property
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
 * @param node - the Node that the pdom behavior is being applied to
 * @param options - options to mutate within the function
 * @param value - the value that you are setting the behavior of, like the accessibleName
 * @param callbacksForOtherNodes - behavior function also support taking state from a Node and using it to
 *   set the accessible content for another Node. If this is the case, that logic should be set in a closure and added to
 *   this list for execution after this Node is fully created. See discussion in https://github.com/phetsims/sun/issues/503#issuecomment-676541373
 *   NOTE: The other Nodes must be a child of this Node, or not in the same subtree. Otherwise, updates could trigger infinite loops in PDOMTree/PDOMPeer update.
 * @returns the options that have been mutated by the behavior function.
 */
type PDOMBehaviorFunction<AllowedKeys extends keyof ParallelDOMOptions> = ( node: Node, options: Pick<ParallelDOMOptions, AllowedKeys>, value: PDOMValueType, callbacksForOtherNodes: ( () => void )[] ) => ParallelDOMOptions;

// Each behavior function supports a limited set of lower level options, as full access to the API in the behavior function can create
// confusing side effects.
export type AccessibleNameBehaviorFunction = PDOMBehaviorFunction<'innerContent' | 'ariaLabel' | 'labelContent' | 'labelTagName' | 'appendLabel'>;
export type AccessibleHelpTextBehaviorFunction = PDOMBehaviorFunction<'descriptionTagName' | 'descriptionContent' | 'appendDescription'>;
export type AccessibleParagraphBehaviorFunction = PDOMBehaviorFunction<'accessibleParagraphContent'>;

export default class ParallelDOM extends PhetioObject {

  // The HTML tag name of the element representing this Node in the DOM
  private _tagName: string | null;

  // The HTML tag name for a container parent element for this Node in the DOM. This
  // container parent will contain the Node's DOM element, as well as peer elements for any label or description
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
  private _inputValue: PDOMValueType | number | null = null;

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

  // Array of attributes that are on the Node's DOM element.  Objects will have the
  // form { attribute:{string}, value:{*}, namespace:{string|null} }
  private _pdomAttributes: PDOMAttribute[];

  // Collection of class attributes that are applied to the Node's DOM element.
  // Objects have the form { className:{string}, options:{*} }
  private _pdomClasses: PDOMClass[];

  // The heading content for the auto-generated accessible heading (or null indicates no heading is generated).
  // See setAccessibleHeading() for more documentation
  private _accessibleHeading: PDOMValueType = null;

  // The label content for this Node's DOM element.  There are multiple ways that a label
  // can be associated with a Node's dom element, see setLabelContent() for more documentation
  private _accessibleHeadingIncrement = 1;

  // The label content for this Node's DOM element.  There are multiple ways that a label
  // can be associated with a Node's dom element, see setLabelContent() for more documentation
  private _labelContent: PDOMValueType = null;

  // The inner label content for this Node's primary sibling. Set as inner HTML
  // or text content of the actual DOM element. If this is used, the Node should not have children.
  private _innerContent: PDOMValueType = null;

  // The description content for this Node's DOM element.
  private _descriptionContent: PDOMValueType = null;

  // The content for the accessible paragraph for this Node's DOM element.
  private _accessibleParagraphContent: PDOMValueType = null;

  // If provided, it will create the primary DOM element with the specified namespace.
  // This may be needed, for example, with MathML/SVG/etc.
  private _pdomNamespace: string | null;

  // If provided, "aria-label" will be added as an inline attribute on the Node's DOM
  // element and set to this value. This will determine how the Accessible Name is provided for the DOM element.
  private _ariaLabel: PDOMValueType = null;
  private _hasAppliedAriaLabel = false;

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
  private _ariaValueText: PDOMValueType = null;
  private _hasAppliedAriaValueText = false;

  // The aria-roledescription assigned to this Node.
  private _accessibleRoleDescription: PDOMValueType = null;

  // Keep track of what this Node is aria-labelledby via "associationObjects"
  // see addAriaLabelledbyAssociation for why we support more than one association.
  private _ariaLabelledbyAssociations: Association[];

  // Keep a reference to all Nodes that are aria-labelledby this Node, i.e. that have store one of this Node's
  // peer HTMLElement's id in their peer HTMLElement's aria-labelledby attribute. This way we can tell other
  // Nodes to update their aria-labelledby associations when this Node rebuilds its pdom content.
  private _nodesThatAreAriaLabelledbyThisNode: Node[];

  // Keep track of what this Node is aria-describedby via "associationObjects"
  // see addAriaDescribedbyAssociation for why we support more than one association.
  private _ariaDescribedbyAssociations: Association[];

  // Keep a reference to all Nodes that are aria-describedby this Node, i.e. that have store one of this Node's
  // peer HTMLElement's id in their peer HTMLElement's aria-describedby attribute. This way we can tell other
  // Nodes to update their aria-describedby associations when this Node rebuilds its pdom content.
  private _nodesThatAreAriaDescribedbyThisNode: Node[];

  // Keep track of what this Node is aria-activedescendant via "associationObjects"
  // see addActiveDescendantAssociation for why we support more than one association.
  private _activeDescendantAssociations: Association[];

  // Keep a reference to all Nodes that are aria-activedescendant this Node, i.e. that have store one of this Node's
  // peer HTMLElement's id in their peer HTMLElement's aria-activedescendant attribute. This way we can tell other
  // Nodes to update their aria-activedescendant associations when this Node rebuilds its pdom content.
  private _nodesThatAreActiveDescendantToThisNode: Node[];

  // Whether this Node's primary sibling has been explicitly set to receive focus from
  // tab navigation. Sets the tabIndex attribute on the Node's primary sibling. Setting to false will not remove the
  // Node's DOM from the document, but will ensure that it cannot receive focus by pressing 'tab'.  Several
  // HTMLElements (such as HTML form elements) can be focusable by default, without setting this property. The
  // native HTML function from these form elements can be overridden with this property.
  private _focusableOverride: boolean | null;

  // The focus highlight that will surround this Node when it
  // is focused.  By default, the focus highlight will be a pink rectangle that surrounds the Node's local
  // bounds. When providing a custom highlight, draw around the Node's local coordinate frame.
  private _focusHighlight: Shape | Node | 'invisible' | null;

  // A flag that allows prevents focus highlight from being displayed in the HighlightOverlay.
  // If true, the focus highlight for this Node will be layerable in the scene graph.  PhetioClient is responsible
  // for placement of the focus highlight in the scene graph.
  private _focusHighlightLayerable: boolean;

  // Adds a group focus highlight that surrounds this Node when a descendant has
  // focus. Typically useful to indicate focus if focus enters a group of elements. If 'true', group
  // highlight will go around local bounds of this Node. Otherwise the custom Node will be used as the highlight/
  private _groupFocusHighlight: Node | boolean;

  // Whether the pdom content will be visible from the browser and assistive
  // technologies.  When pdomVisible is false, the Node's primary sibling will not be focusable, and it cannot
  // be found by the assistive technology virtual cursor. For more information on how assistive technologies
  // read with the virtual cursor see
  // http://www.ssbbartgroup.com/blog/how-windows-screen-readers-work-on-the-web/
  private readonly _pdomVisibleProperty: TinyForwardingProperty<boolean>;

  // If provided, it will override the focus order between children
  // (and optionally arbitrary subtrees). If not provided, the focus order will default to the rendering order
  // (first children first, last children last) determined by the children array.
  // See setPDOMOrder() for more documentation.
  private _pdomOrder: ( Node | null )[] | null;

  // If this Node is specified in another Node's pdomOrder, then this will have the value of that other (PDOM parent)
  // Node. Otherwise it's null.
  // (scenery-internal)
  public _pdomParent: Node | null;

  // If this is specified, the primary sibling will be positioned
  // to align with this source Node and observe the transforms along this Node's trail. At this time the
  // pdomTransformSourceNode cannot use DAG.
  private _pdomTransformSourceNode: Node | null;

  // If this is provided, the AnimatedPanZoomListener will attempt to keep this Node in view as long as it has
  // focus
  private _focusPanTargetBoundsProperty: TReadOnlyProperty<Bounds2> | null;

  // If provided, the AnimatedPanZoomListener will ONLY pan in the specified direction
  private _limitPanDirection: LimitPanDirection | null;

  // Contains information about what pdom displays
  // this Node is "visible" for, see PDOMDisplaysInfo.js for more information.
  // (scenery-internal)
  public _pdomDisplaysInfo: PDOMDisplaysInfo;

  // Empty unless the Node contains some pdom content (PDOMInstance).
  private readonly _pdomInstances: PDOMInstance[];

  // Determines if DOM siblings are positioned in the viewport. This
  // is required for Nodes that require unique input gestures with iOS VoiceOver like "Drag and Drop".
  // See setPositionInPDOM for more information.
  private _positionInPDOM: boolean;

  // If true, any DOM input events received from the label sibling will not be dispatched as SceneryEvents in Input.js.
  // The label sibling may receive input by screen readers if the virtual cursor is over it. That is usually fine,
  // but there is a bug with NVDA and Firefox where both the label sibling AND primary sibling receive events in
  // this case, and both bubble up to the root of the PDOM, and so we would otherwise dispatch two SceneryEvents
  // instead of one.
  private _excludeLabelSiblingFromInput: boolean;

  // HIGHER LEVEL API INITIALIZATION

  // Sets the "Accessible Name" of the Node, as defined by the Browser's ParallelDOM Tree
  private _accessibleName: PDOMValueType = null;

  // Function that returns the options needed to set the appropriate accessible name for the Node
  private _accessibleNameBehavior: AccessibleNameBehaviorFunction;

  // Sets the 'Accessible Paragraph' for the Node. This makes this Node a paragraph of descriptive content, often
  // for non-interactive elements.
  private _accessibleParagraph: PDOMValueType = null;

  // Function that returns the options needed to set the appropriate accessible paragraph for the Node.
  private _accessibleParagraphBehavior: AccessibleParagraphBehaviorFunction;

  // Sets the help text of the Node, this most often corresponds to description text.
  private _accessibleHelpText: PDOMValueType = null;

  // Sets the help text of the Node, this most often corresponds to description text.
  private _accessibleHelpTextBehavior: AccessibleHelpTextBehaviorFunction;

  // Forces an update from the behavior functions in PDOMPeer.
  // (scenery-internal)
  public _accessibleNameDirty = false;
  public _accessibleHelpTextDirty = false;
  public _accessibleParagraphDirty = false;

  // Emits an event when the focus highlight is changed.
  public readonly focusHighlightChangedEmitter: TEmitter = new TinyEmitter();

  // Emits an event when the pdom parent of this Node has changed
  public readonly pdomParentChangedEmitter: TEmitter = new TinyEmitter();

  // Fired when the PDOM Displays for this Node have changed (see PDOMInstance)
  public readonly pdomDisplaysEmitter: TEmitter = new TinyEmitter();

  // PDOM specific enabled listener
  protected pdomBoundInputEnabledListener: ( enabled: boolean ) => void;

  // Used to make sure that we do not recursively create PDOMInstances when doing PDOMTree operations.
  // (scenery-internal)
  public _lockedPDOMInstanceCreation = false;

  protected _onPDOMContentChangeListener: () => void;
  protected _onInputValueChangeListener: () => void;
  protected _onAriaLabelChangeListener: () => void;
  protected _onAccessibleRoleDescriptionChangeListener: () => void;
  protected _onAriaValueTextChangeListener: () => void;
  protected _onLabelContentChangeListener: () => void;
  protected _onAccessibleHeadingChangeListener: () => void;
  protected _onDescriptionContentChangeListener: () => void;
  protected _onAccessibleParagraphContentChangeListener: () => void;
  protected _onInnerContentChangeListener: () => void;

  protected constructor( options?: PhetioObjectOptions ) {

    super( options );

    this._onPDOMContentChangeListener = this.onPDOMContentChange.bind( this );
    this._onInputValueChangeListener = this.invalidatePeerInputValue.bind( this );
    this._onAriaLabelChangeListener = this.onAriaLabelChange.bind( this );
    this._onAriaValueTextChangeListener = this.onAriaValueTextChange.bind( this );
    this._onLabelContentChangeListener = this.invalidatePeerLabelSiblingContent.bind( this );
    this._onAccessibleHeadingChangeListener = this.invalidateAccessibleHeadingContent.bind( this );
    this._onDescriptionContentChangeListener = this.invalidatePeerDescriptionSiblingContent.bind( this );
    this._onAccessibleParagraphContentChangeListener = this.invalidatePeerParagraphSiblingContent.bind( this );
    this._onInnerContentChangeListener = this.onInnerContentPropertyChange.bind( this );
    this._onAccessibleRoleDescriptionChangeListener = this.onAccessibleRoleDescriptionChange.bind( this );

    this._tagName = null;
    this._containerTagName = null;
    this._labelTagName = null;
    this._descriptionTagName = null;
    this._inputType = null;
    this._pdomChecked = false;
    this._appendLabel = false;
    this._appendDescription = false;
    this._pdomAttributes = [];
    this._pdomClasses = [];

    this._pdomNamespace = null;
    this._ariaRole = null;
    this._containerAriaRole = null;
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
    this._pdomOrder = null;
    this._pdomParent = null;
    this._pdomTransformSourceNode = null;
    this._focusPanTargetBoundsProperty = null;
    this._limitPanDirection = null;
    this._pdomDisplaysInfo = new PDOMDisplaysInfo( this as unknown as Node );
    this._pdomInstances = [];
    this._positionInPDOM = false;
    this._excludeLabelSiblingFromInput = false;

    this._pdomVisibleProperty = new TinyForwardingProperty<boolean>( true, false, this.onPdomVisiblePropertyChange.bind( this ) );

    // HIGHER LEVEL API INITIALIZATION

    this._accessibleNameBehavior = ParallelDOM.BASIC_ACCESSIBLE_NAME_BEHAVIOR;
    this._accessibleHelpTextBehavior = ParallelDOM.HELP_TEXT_AFTER_CONTENT;
    this._accessibleParagraphBehavior = ParallelDOM.BASIC_ACCESSIBLE_PARAGRAPH_BEHAVIOR;
    this.pdomBoundInputEnabledListener = this.pdomInputEnabledListener.bind( this );
  }

  /***********************************************************************************************************/
  // PUBLIC METHODS
  /***********************************************************************************************************/

  /**
   * Dispose accessibility by removing all listeners on this Node for accessible input. ParallelDOM is disposed
   * by calling Node.dispose(), so this function is scenery-internal.
   * (scenery-internal)
   */
  protected disposeParallelDOM(): void {

    if ( isTReadOnlyProperty( this._accessibleName ) && !this._accessibleName.isDisposed ) {
      this._accessibleName.unlink( this._onPDOMContentChangeListener );
      this._accessibleName = null;
    }

    if ( isTReadOnlyProperty( this._accessibleHelpText ) && !this._accessibleHelpText.isDisposed ) {
      this._accessibleHelpText.unlink( this._onPDOMContentChangeListener );
      this._accessibleHelpText = null;
    }

    if ( isTReadOnlyProperty( this._accessibleParagraph ) && !this._accessibleParagraph.isDisposed ) {
      this._accessibleParagraph.unlink( this._onPDOMContentChangeListener );
      this._accessibleParagraph = null;
    }

    if ( isTReadOnlyProperty( this._inputValue ) && !this._inputValue.isDisposed ) {
      this._inputValue.unlink( this._onPDOMContentChangeListener );
      this._inputValue = null;
    }

    if ( isTReadOnlyProperty( this._ariaLabel ) && !this._ariaLabel.isDisposed ) {
      this._ariaLabel.unlink( this._onAriaLabelChangeListener );
    }

    if ( isTReadOnlyProperty( this._ariaValueText ) && !this._ariaValueText.isDisposed ) {
      this._ariaValueText.unlink( this._onAriaValueTextChangeListener );
    }

    if ( isTReadOnlyProperty( this._accessibleRoleDescription ) && !this._accessibleRoleDescription.isDisposed ) {
      this._accessibleRoleDescription.unlink( this._onAccessibleRoleDescriptionChangeListener );
    }

    if ( isTReadOnlyProperty( this._innerContent ) && !this._innerContent.isDisposed ) {
      this._innerContent.unlink( this._onInnerContentChangeListener );
    }

    if ( isTReadOnlyProperty( this._labelContent ) && !this._labelContent.isDisposed ) {
      this._labelContent.unlink( this._onLabelContentChangeListener );
    }

    if ( isTReadOnlyProperty( this._accessibleHeading ) && !this._accessibleHeading.isDisposed ) {
      this._accessibleHeading.unlink( this._onAccessibleHeadingChangeListener );
    }

    if ( isTReadOnlyProperty( this._descriptionContent ) && !this._descriptionContent.isDisposed ) {
      this._descriptionContent.unlink( this._onDescriptionContentChangeListener );
    }

    if ( isTReadOnlyProperty( this._accessibleParagraphContent ) && !this._accessibleParagraphContent.isDisposed ) {
      this._accessibleParagraphContent.unlink( this._onAccessibleParagraphContentChangeListener );
    }

    ( this as unknown as Node ).inputEnabledProperty.unlink( this.pdomBoundInputEnabledListener );

    // To prevent memory leaks, we want to clear our order (since otherwise Nodes in our order will reference
    // this Node).
    this.pdomOrder = null;

    // If this Node is in any PDOM order, we need to remove it from the order of the other Node so there is
    // no reference to this Node.
    if ( this._pdomParent ) {
      assert && assert( this._pdomParent._pdomOrder, 'pdomParent should have a pdomOrder' );
      const updatedOrder = this._pdomParent._pdomOrder!.slice();
      arrayRemove( updatedOrder, this as unknown as Node );
      this._pdomParent.pdomOrder = updatedOrder;
    }

    // clear references to the pdomTransformSourceNode
    this.setPDOMTransformSourceNode( null );

    // Clear behavior functions because they may create references between other Nodes
    this._accessibleNameBehavior = ParallelDOM.BASIC_ACCESSIBLE_NAME_BEHAVIOR;
    this._accessibleHelpTextBehavior = ParallelDOM.HELP_TEXT_AFTER_CONTENT;

    // Clear out aria association attributes, which hold references to other Nodes.
    this.setAriaLabelledbyAssociations( [] );
    this.setAriaDescribedbyAssociations( [] );
    this.setActiveDescendantAssociations( [] );

    // PDOM attributes can potentially have listeners, so we will clear those out.
    this.removePDOMAttributes();

    this._pdomVisibleProperty.dispose();
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
   * Focus this Node's primary dom element. The element must not be hidden, and it must be focusable. If the Node
   * has more than one instance, this will fail because the DOM element is not uniquely defined. If accessibility
   * is not enabled, this will be a no op. When ParallelDOM is more widely used, the no op can be replaced
   * with an assertion that checks for pdom content.
   */
  public focus(): void {

    // if a sim is running without accessibility enabled, there will be no accessible instances, but focus() might
    // still be called without accessibility enabled
    if ( this._pdomInstances.length > 0 ) {

      // when accessibility is widely used, this assertion can be added back in
      // assert && assert( this._pdomInstances.length > 0, 'there must be pdom content for the Node to receive focus' );
      assert && assert( this.focusable, 'trying to set focus on a Node that is not focusable' );
      assert && assert( this.pdomVisible, 'trying to set focus on a Node with invisible pdom content' );
      assert && assert( this._pdomInstances.length === 1, 'focus() unsupported for Nodes using DAG, pdom content is not unique' );

      const peer = this._pdomInstances[ 0 ].peer!;
      assert && assert( peer, 'must have a peer to focus' );
      peer.focus();
    }
  }

  /**
   * Remove focus from this Node's primary DOM element.  The focus highlight will disappear, and the element will not receive
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

    if ( assert && this.hasPDOMContent ) {

      if ( this._tagName ) {
        this._inputType && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support inputType' );
        this._pdomChecked && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support pdomChecked.' );
        this._inputValue && assert( this._tagName.toUpperCase() === INPUT_TAG, 'tagName must be INPUT to support inputValue' );

        this._pdomChecked && assert( INPUT_TYPES_THAT_SUPPORT_CHECKED.includes( this._inputType!.toUpperCase() ), `inputType does not support checked attribute: ${this._inputType}` );
        this._focusHighlightLayerable && assert( this.focusHighlight instanceof ParallelDOM, 'focusHighlight must be Node if highlight is layerable' );
        this._tagName.toUpperCase() === INPUT_TAG && assert( typeof this._inputType === 'string', ' inputType expected for input' );
      }

      // note that most things that are not focusable by default need innerContent to be focusable on VoiceOver,
      // but this will catch most cases since often things that get added to the focus order have the application
      // role for custom input. Note that accessibleName will not be checked that it specifically changes innerContent, it is up to the dev to do this.
      this.ariaRole === 'application' && this.focusable && assert( this.innerContent || this.accessibleName, 'must have some innerContent or element will never be focusable in VoiceOver' );

      // If using accessibleParagraph without a tagName, this Node cannot have any descendants with accessible content
      if ( this.accessibleParagraph && !this.tagName ) {
        this.pdomInstances.forEach( pdomInstance => {
          assert && assert( pdomInstance.children.length === 0, 'Assign a tagName to a Node if it has descendants with accessible content.' );
        } );
      }
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
   * Sets the accessible name that describes this Node. The accessible name is the semantic title for the Node. It is
   * the content that will be read by a screen reader when the Node is discovered by the virtual cursor.
   *
   * For more information about accessible names in web accessibility see
   * https://developer.paciellogroup.com/blog/2017/04/what-is-an-accessible-name/.
   *
   * Part of the higher level API, the accessibleNameBehavior function will set the appropriate options on this Node
   * to create the desired accessible name. See the documentation for setAccessibleNameBehavior() for more information.
   */
  public setAccessibleName( accessibleName: PDOMValueType ): void {
    if ( accessibleName !== this._accessibleName ) {
      if ( isTReadOnlyProperty( this._accessibleName ) && !this._accessibleName.isDisposed ) {
        this._accessibleName.unlink( this._onPDOMContentChangeListener );
      }

      this._accessibleName = accessibleName;

      if ( isTReadOnlyProperty( accessibleName ) ) {
        accessibleName.lazyLink( this._onPDOMContentChangeListener );
      }

      this._accessibleNameDirty = true;

      this.onPDOMContentChange();
    }
  }

  public set accessibleName( accessibleName: PDOMValueType ) { this.setAccessibleName( accessibleName ); }

  public get accessibleName(): string | null { return this.getAccessibleName(); }

  /**
   * Get the accessible name that describes this Node.
   */
  public getAccessibleName(): string | null {
    if ( isTReadOnlyProperty( this._accessibleName ) ) {
      return this._accessibleName.value;
    }
    else {
      return this._accessibleName;
    }
  }

  /**
   * Sets content for a paragraph that describes this Node for screen readers. This
   * is most useful for non-interactive elements that need to be described.
   *
   * For example:
   * myImageNode.setAccessibleParagraph( 'This is a picture of a cat' );
   */
  public setAccessibleParagraph( accessibleParagraph: PDOMValueType ): void {
    if ( accessibleParagraph !== this._accessibleParagraph ) {

      if ( isTReadOnlyProperty( this._accessibleParagraph ) && !this._accessibleParagraph.isDisposed ) {
        this._accessibleParagraph.unlink( this._onPDOMContentChangeListener );
      }

      this._accessibleParagraph = accessibleParagraph;

      if ( isTReadOnlyProperty( accessibleParagraph ) ) {
        accessibleParagraph.lazyLink( this._onPDOMContentChangeListener );
      }

      this._accessibleParagraphDirty = true;

      // The behavior function may change any of this Node's state so we need to recompute all content.
      this.onPDOMContentChange();
    }
  }

  public set accessibleParagraph( accessibleParagraph: PDOMValueType ) { this.setAccessibleParagraph( accessibleParagraph ); }

  public get accessibleParagraph(): string | null { return this.getAccessibleParagraph(); }

  /**
   * Get the accessible paragraph that represents/describe.
   */
  public getAccessibleParagraph(): string | null {
    if ( isTReadOnlyProperty( this._accessibleParagraph ) ) {
      return this._accessibleParagraph.value;
    }
    else {
      return this._accessibleParagraph;
    }
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
   * accessibleNameBehavior is a function that will set the appropriate options on this Node to get the desired
   * accessible name.
   *
   * The default value does the best it can to create an accessible name for a variety of different ParallelDOM
   * options and tag names. If a Node uses more complicated markup, you can provide your own function to
   * meet your requirements. If you do this, it is up to you to make sure that the Accessible Name is properly
   * being set and conveyed to AT, as it is very hard to validate this function.
   */
  public setAccessibleNameBehavior( accessibleNameBehavior: AccessibleNameBehaviorFunction ): void {

    if ( this._accessibleNameBehavior !== accessibleNameBehavior ) {

      this._accessibleNameBehavior = accessibleNameBehavior;

      this.onPDOMContentChange();
    }
  }

  public set accessibleNameBehavior( accessibleNameBehavior: AccessibleNameBehaviorFunction ) { this.setAccessibleNameBehavior( accessibleNameBehavior ); }

  public get accessibleNameBehavior(): AccessibleNameBehaviorFunction { return this.getAccessibleNameBehavior(); }

  /**
   * Get the help text of the interactive element.
   */
  public getAccessibleNameBehavior(): AccessibleNameBehaviorFunction {
    return this._accessibleNameBehavior;
  }

  /**
   * Sets the accessible help text for this Node. Help text usually provides additional information that describes
   * what a Node is or how to interact with it. It will be read by a screen reader when discovered by the virtual
   * cursor.
   *
   * Part of the higher level API, the accessibleHelpTextBehavior function will set the appropriate options on this Node
   * to create the desired help text. See the documentation for setAccessibleHelpTextBehavior() for more information.
   */
  public setAccessibleHelpText( accessibleHelpText: PDOMValueType ): void {
    if ( accessibleHelpText !== this._accessibleHelpText ) {
      if ( isTReadOnlyProperty( this._accessibleHelpText ) && !this._accessibleHelpText.isDisposed ) {
        this._accessibleHelpText.unlink( this._onPDOMContentChangeListener );
      }

      this._accessibleHelpText = accessibleHelpText;

      if ( isTReadOnlyProperty( accessibleHelpText ) ) {
        accessibleHelpText.lazyLink( this._onPDOMContentChangeListener );
      }

      this._accessibleHelpTextDirty = true;

      this.onPDOMContentChange();
    }
  }

  public set accessibleHelpText( accessibleHelpText: PDOMValueType ) { this.setAccessibleHelpText( accessibleHelpText ); }

  public get accessibleHelpText(): string | null { return this.getAccessibleHelpText(); }

  /**
   * Get the help text for this Node.
   */
  public getAccessibleHelpText(): string | null {
    if ( isTReadOnlyProperty( this._accessibleHelpText ) ) {
      return this._accessibleHelpText.value;
    }
    else {
      return this._accessibleHelpText;
    }
  }

  /**
   * Sets the accessible paragraph content for this Node. This is a paragraph of descriptive content, often
   * for non-interactive elements.
   *
   * This is the lower level API function for accessibleParagraph. You probably just want to use setAccessibleParagraph
   * instead.
   */
  public setAccessibleParagraphContent( accessibleParagraphContent: PDOMValueType ): void {
    if ( accessibleParagraphContent !== this._accessibleParagraphContent ) {
      if ( isTReadOnlyProperty( this._accessibleParagraphContent ) && !this._accessibleParagraphContent.isDisposed ) {
        this._accessibleParagraphContent.unlink( this._onAccessibleParagraphContentChangeListener );
      }

      this._accessibleParagraphContent = accessibleParagraphContent;

      if ( isTReadOnlyProperty( accessibleParagraphContent ) ) {
        accessibleParagraphContent.lazyLink( this._onAccessibleParagraphContentChangeListener );
      }

      this.invalidatePeerParagraphSiblingContent();
    }
  }

  public set accessibleParagraphContent( content: PDOMValueType ) { this.setAccessibleParagraphContent( content ); }

  /**
   * Returns the accessibleParagraph content for this Node.
   */
  public get accessibleParagraphContent(): string | null { return this.getAccessibleParagraphContent(); }

  public getAccessibleParagraphContent(): string | null {
    return unwrapProperty( this._accessibleParagraphContent );
  }


  /**
   * Sets the accessible heading for this Node. If non-null, a heading element (e.g. <h3>) will be created and will have
   * text content equal to this value. It will be read by a screen reader when discovered by the virtual cursor.
   *
   * By default, the heading level (h1 ... h6) that is chosen will be one greater than the heading level of the closest
   * ancestor. The base level is set on the Display with `baseHeadingLevel` (which can be useful if embedding the Display
   * within another document that has outside headings.
   *
   * If the heading level REQUIRES adjustment (due to node structure that cannot be easily changed), it is possible to
   * modify the heading level, see accessibleHeadingIncrement.
   *
   * Another way to adjust the computed heading level is to just use low level API options (labelTagName
   * and labelContent). The computed heading level does not consider Nodes with headings defined with
   * labelTagName or other options.
   *
   * This method supports adding content in two ways, with HTMLElement.textContent and HTMLElement.innerHTML.
   * The DOM setter is chosen based on if the label passes the `containsFormattingTags`.
   */
  public setAccessibleHeading( accessibleHeading: PDOMValueType ): void {
    if ( accessibleHeading !== this._accessibleHeading ) {
      const headingExistenceChanged = ( accessibleHeading === null ) !== ( this._accessibleHeading === null );

      if ( isTReadOnlyProperty( this._accessibleHeading ) && !this._accessibleHeading.isDisposed ) {
        this._accessibleHeading.unlink( this._onAccessibleHeadingChangeListener );
      }

      this._accessibleHeading = accessibleHeading;

      if ( isTReadOnlyProperty( accessibleHeading ) ) {
        accessibleHeading.lazyLink( this._onAccessibleHeadingChangeListener );
      }

      if ( headingExistenceChanged ) {
        this.invalidateAccessibleHeading();
      }
      else {
        this.invalidateAccessibleHeadingContent();
      }
    }
  }

  public set accessibleHeading( label: PDOMValueType ) { this.setAccessibleHeading( label ); }

  public get accessibleHeading(): string | null { return this.getAccessibleHeading(); }

  /**
   * Get the value of this Node's accessibleHeading. See setAccessibleHeading() for more information.
   */
  public getAccessibleHeading(): string | null {
    return unwrapProperty( this._accessibleHeading );
  }

  /**
   * Sets the heading increment. THIS IS AN OVERRIDE, please instead adjust the node structure so this isn't needed!
   *
   * The heading level chosen for this node will be `nearestParentHeadingLevel + accessibleHeadingIncrement`, where
   * nearestParentHeadingLevel is found on a parent, or is determined by the default `baseHeadingLevel` on the `Display`.
   *
   * The default is 1 (so that nesting headings results in a natural structure, increasing the level).
   *
   * For cases where there is a parent-child relationship between two nodes that both have headings, BUT it is desired
   * for them to have the same heading level (e.g. both <h3>), the accessibleHeadingIncrement can be set to 0.
   *
   * For cases where there is a sibling relationship between two nodes that both have headings, BUT it is desired for them
   * to have different heading levels (as if they were a parent-child relationship), you can set the accessibleHeadingIncrement
   * on the child-like node to 2.
   */
  public setAccessibleHeadingIncrement( accessibleHeadingIncrement: number ): void {
    if ( accessibleHeadingIncrement !== this._accessibleHeadingIncrement ) {
      this._accessibleHeadingIncrement = accessibleHeadingIncrement;

      this.invalidateAccessibleHeading();
    }
  }

  public set accessibleHeadingIncrement( accessibleHeadingIncrement: number ) {
    this.setAccessibleHeadingIncrement( accessibleHeadingIncrement );
  }

  public get accessibleHeadingIncrement(): number { return this.getAccessibleHeadingIncrement(); }

  /**
   * Get the content for this Node's label sibling DOM element.
   */
  public getAccessibleHeadingIncrement(): number {
    return this._accessibleHeadingIncrement;
  }

  private invalidateAccessibleHeading(): void {
    this.onPDOMContentChange();

    this.invalidateAccessibleHeadingContent();
  }

  private invalidateAccessibleHeadingContent(): void {
    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      const peer = this._pdomInstances[ i ].peer!;
      peer.setHeadingContent( unwrapProperty( this._accessibleHeading ) );
    }
  }

  /**
   * accessibleHelpTextBehavior is a function that will set the appropriate options on this Node to get the desired help text.
   *
   * The default value does the best it can to create the help text based on the values for other ParallelDOM options.
   * Usually, this is a paragraph element that comes after the Node's primary sibling in the PDOM. If you need to
   * customize this behavior, you can provide your own function to meet your requirements. If you provide your own
   * function, it is up to you to make sure that the help text is properly being set and is discoverable by AT.
   */
  public setAccessibleHelpTextBehavior( accessibleHelpTextBehavior: AccessibleHelpTextBehaviorFunction ): void {

    if ( this._accessibleHelpTextBehavior !== accessibleHelpTextBehavior ) {

      this._accessibleHelpTextBehavior = accessibleHelpTextBehavior;

      this.onPDOMContentChange();
    }
  }

  public set accessibleHelpTextBehavior( accessibleHelpTextBehavior: AccessibleHelpTextBehaviorFunction ) { this.setAccessibleHelpTextBehavior( accessibleHelpTextBehavior ); }

  public get accessibleHelpTextBehavior(): AccessibleHelpTextBehaviorFunction { return this.getAccessibleHelpTextBehavior(); }

  /**
   * Get the help text of the interactive element.
   */
  public getAccessibleHelpTextBehavior(): AccessibleHelpTextBehaviorFunction {
    return this._accessibleHelpTextBehavior;
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

      // TODO: this could be setting PDOM content twice https://github.com/phetsims/scenery/issues/1581
      this.onPDOMContentChange();
    }
  }

  public set tagName( tagName: string | null ) { this.setTagName( tagName ); }

  public get tagName(): string | null { return this.getTagName(); }

  /**
   * Get the tag name of the DOM element representing this Node for accessibility.
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

  private invalidatePeerLabelSiblingContent(): void {
    const labelContent = this.labelContent;

    // if trying to set labelContent, make sure that there is a labelTagName default
    if ( labelContent && !this._labelTagName ) {
      this.setLabelTagName( DEFAULT_LABEL_TAG_NAME );
    }

    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      const peer = this._pdomInstances[ i ].peer!;
      peer.setLabelSiblingContent( labelContent );
    }
  }

  /**
   * Set the content of the label sibling for the this Node.  The label sibling will default to the value of
   * DEFAULT_LABEL_TAG_NAME if no `labelTagName` is provided. If the label sibling is a `LABEL` html element,
   * then the `for` attribute will automatically be added, pointing to the Node's primary sibling.
   *
   * This method supports adding content in two ways, with HTMLElement.textContent and HTMLElement.innerHTML.
   * The DOM setter is chosen based on if the label passes the `containsFormattingTags`.
   *
   * Passing a null label value will not clear the whole label sibling, just the inner content of the DOM Element.
   */
  public setLabelContent( labelContent: PDOMValueType ): void {
    if ( labelContent !== this._labelContent ) {
      if ( isTReadOnlyProperty( this._labelContent ) && !this._labelContent.isDisposed ) {
        this._labelContent.unlink( this._onLabelContentChangeListener );
      }

      this._labelContent = labelContent;

      if ( isTReadOnlyProperty( labelContent ) ) {
        labelContent.lazyLink( this._onLabelContentChangeListener );
      }

      this.invalidatePeerLabelSiblingContent();
    }
  }

  public set labelContent( label: PDOMValueType ) { this.setLabelContent( label ); }

  public get labelContent(): string | null { return this.getLabelContent(); }

  /**
   * Get the content for this Node's label sibling DOM element.
   */
  public getLabelContent(): string | null {
    return unwrapProperty( this._labelContent );
  }

  private onInnerContentPropertyChange(): void {
    const value = this.innerContent;

    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      const peer = this._pdomInstances[ i ].peer!;
      peer.setPrimarySiblingContent( value );
    }
  }

  /**
   * Set the inner content for the primary sibling of the PDOMPeers of this Node. Will be set as textContent
   * unless content is html which uses exclusively formatting tags. A Node with inner content cannot
   * have accessible descendants because this content will override the HTML of descendants of this Node.
   */
  public setInnerContent( innerContent: PDOMValueType ): void {
    if ( innerContent !== this._innerContent ) {
      if ( isTReadOnlyProperty( this._innerContent ) && !this._innerContent.isDisposed ) {
        this._innerContent.unlink( this._onInnerContentChangeListener );
      }

      this._innerContent = innerContent;

      if ( isTReadOnlyProperty( innerContent ) ) {
        innerContent.lazyLink( this._onInnerContentChangeListener );
      }

      this.onInnerContentPropertyChange();
    }
  }

  public set innerContent( content: PDOMValueType ) { this.setInnerContent( content ); }

  public get innerContent(): string | null { return this.getInnerContent(); }

  /**
   * Get the inner content, the string that is the innerHTML or innerText for the Node's primary sibling.
   */
  public getInnerContent(): string | null {
    return unwrapProperty( this._innerContent );
  }

  private invalidatePeerDescriptionSiblingContent(): void {
    const descriptionContent = this.descriptionContent;

    // if there is no description element, assume that a paragraph element should be used
    if ( descriptionContent && !this._descriptionTagName ) {
      this.setDescriptionTagName( DEFAULT_DESCRIPTION_TAG_NAME );
    }

    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      const peer = this._pdomInstances[ i ].peer!;
      peer.setDescriptionSiblingContent( descriptionContent );
    }
  }

  /**
   * When the accessible paragraph changes, this will update content in the ParallelDOM by forwarding
   * the update change to the PDOMPeer.
   */
  private invalidatePeerParagraphSiblingContent(): void {
    for ( let i = 0; i < this._pdomInstances.length; i++ ) {
      const peer = this._pdomInstances[ i ].peer!;
      peer.setAccessibleParagraphContent( this.accessibleParagraphContent );
    }
  }

  /**
   * Set the description content for this Node's primary sibling. The description sibling tag name must support
   * innerHTML and textContent. If a description element does not exist yet, a default
   * DEFAULT_LABEL_TAG_NAME will be assigned to the descriptionTagName.
   */
  public setDescriptionContent( descriptionContent: PDOMValueType ): void {
    if ( descriptionContent !== this._descriptionContent ) {
      if ( isTReadOnlyProperty( this._descriptionContent ) && !this._descriptionContent.isDisposed ) {
        this._descriptionContent.unlink( this._onDescriptionContentChangeListener );
      }

      this._descriptionContent = descriptionContent;

      if ( isTReadOnlyProperty( descriptionContent ) ) {
        descriptionContent.lazyLink( this._onDescriptionContentChangeListener );
      }

      this.invalidatePeerDescriptionSiblingContent();
    }
  }

  public set descriptionContent( textContent: PDOMValueType ) { this.setDescriptionContent( textContent ); }

  public get descriptionContent(): string | null { return this.getDescriptionContent(); }

  /**
   * Get the content for this Node's description sibling DOM Element.
   */
  public getDescriptionContent(): string | null {
    return unwrapProperty( this._descriptionContent );
  }

  /**
   * Set the accessibleParagraph behavior function which can be used to control the implementation of the accessbibleParagraph.
   */
  public setAccessibleParagraphBehavior( accessibleParagraphBehavior: AccessibleParagraphBehaviorFunction ): void {
    if ( this._accessibleParagraphBehavior !== accessibleParagraphBehavior ) {

      this._accessibleParagraphBehavior = accessibleParagraphBehavior;

      this.onPDOMContentChange();
    }
  }

  public set accessibleParagraphBehavior( accessibleParagraphBehavior: AccessibleParagraphBehaviorFunction ) { this.setAccessibleParagraphBehavior( accessibleParagraphBehavior ); }

  public get accessibleParagraphBehavior(): AccessibleParagraphBehaviorFunction { return this.getAccessibleParagraphBehavior(); }

  public getAccessibleParagraphBehavior(): AccessibleParagraphBehaviorFunction {
    return this._accessibleParagraphBehavior;
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
   * Get the ARIA role representing this Node.
   */
  public getAriaRole(): string | null {
    return this._ariaRole;
  }

  /**
   * Set the ARIA role for this Node's container parent element.  According to the W3C, the ARIA role is read-only
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
          elementName: PEER_CONTAINER_PARENT
        } );
      }

      // add the attribute
      else {
        this.setPDOMAttribute( 'role', ariaRole, {
          elementName: PEER_CONTAINER_PARENT
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

  private onAriaValueTextChange(): void {
    const ariaValueText = this.ariaValueText;

    if ( ariaValueText === null ) {
      if ( this._hasAppliedAriaLabel ) {
        this.removePDOMAttribute( 'aria-valuetext' );
        this._hasAppliedAriaLabel = false;
      }
    }
    else {
      this.setPDOMAttribute( 'aria-valuetext', ariaValueText );
      this._hasAppliedAriaLabel = true;
    }
  }

  /**
   * Updates the attribute value whenever there is a change to the aria-roledescription.
   */
  private onAccessibleRoleDescriptionChange(): void {
    const accessibleRoleDescription = this.accessibleRoleDescription;
    if ( accessibleRoleDescription === null ) {
      this.removePDOMAttribute( 'aria-roledescription' );
    }
    else {
      this.setPDOMAttribute( 'aria-roledescription', accessibleRoleDescription );
    }
  }

  /**
   * Set the aria-valuetext of this Node independently from the changing value, if necessary. Setting to null will
   * clear this attribute.
   */
  public setAriaValueText( ariaValueText: PDOMValueType ): void {
    if ( this._ariaValueText !== ariaValueText ) {
      if ( isTReadOnlyProperty( this._ariaValueText ) && !this._ariaValueText.isDisposed ) {
        this._ariaValueText.unlink( this._onAriaValueTextChangeListener );
      }

      this._ariaValueText = ariaValueText;

      if ( isTReadOnlyProperty( ariaValueText ) ) {
        ariaValueText.lazyLink( this._onAriaValueTextChangeListener );
      }

      this.onAriaValueTextChange();
    }
  }

  public set ariaValueText( ariaValueText: PDOMValueType ) { this.setAriaValueText( ariaValueText ); }

  public get ariaValueText(): string | null { return this.getAriaValueText(); }

  /**
   * Get the value of the aria-valuetext attribute for this Node's primary sibling. If null, then the attribute
   * has not been set on the primary sibling.
   */
  public getAriaValueText(): string | null {
    return unwrapProperty( this._ariaValueText );
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

  private onAriaLabelChange(): void {
    const ariaLabel = this.ariaLabel;

    if ( ariaLabel === null ) {
      if ( this._hasAppliedAriaLabel ) {
        this.removePDOMAttribute( 'aria-label' );
        this._hasAppliedAriaLabel = false;
      }
    }
    else {
      this.setPDOMAttribute( 'aria-label', ariaLabel );
      this._hasAppliedAriaLabel = true;
    }
  }

  /**
   * Sets the 'aria-label' attribute for labelling the Node's primary sibling. By using the
   * 'aria-label' attribute, the label will be read on focus, but can not be found with the
   * virtual cursor. This is one way to set a DOM Element's Accessible Name.
   *
   * @param ariaLabel - the text for the aria label attribute
   */
  public setAriaLabel( ariaLabel: PDOMValueType ): void {
    if ( this._ariaLabel !== ariaLabel ) {
      if ( isTReadOnlyProperty( this._ariaLabel ) && !this._ariaLabel.isDisposed ) {
        this._ariaLabel.unlink( this._onAriaLabelChangeListener );
      }

      this._ariaLabel = ariaLabel;

      if ( isTReadOnlyProperty( ariaLabel ) ) {
        ariaLabel.lazyLink( this._onAriaLabelChangeListener );
      }

      this.onAriaLabelChange();
    }
  }

  public set ariaLabel( ariaLabel: PDOMValueType ) { this.setAriaLabel( ariaLabel ); }

  public get ariaLabel(): string | null { return this.getAriaLabel(); }

  /**
   * Get the value of the aria-label attribute for this Node's primary sibling.
   */
  public getAriaLabel(): string | null {
    return unwrapProperty( this._ariaLabel );
  }

  /**
   * Sets an aria-roledescription for this Node, describing its interactive purpose and user
   * interaction methods.
   *
   * Use sparingly, and avoid overriding standard roles. This is especially helpful for
   * unique or unconventional UI components.
   *
   * This function works by adding aria-roledescription to this Node's list of PDOM attributes.
   */
  public setAccessibleRoleDescription( roleDescription: PDOMValueType ): void {
    if ( this._accessibleRoleDescription !== roleDescription ) {
      if ( isTReadOnlyProperty( this._accessibleRoleDescription ) && !this._accessibleRoleDescription.isDisposed ) {
        this._accessibleRoleDescription.unlink( this._onAccessibleRoleDescriptionChangeListener );
      }

      this._accessibleRoleDescription = roleDescription;

      if ( isTReadOnlyProperty( roleDescription ) ) {
        roleDescription.lazyLink( this._onAccessibleRoleDescriptionChangeListener );
      }

      this.onAccessibleRoleDescriptionChange();
    }
  }

  public set accessibleRoleDescription( roleDescription: PDOMValueType ) { this.setAccessibleRoleDescription( roleDescription ); }

  public get accessibleRoleDescription(): string | null { return this.getAccessibleRoleDescription(); }

  public getAccessibleRoleDescription(): string | null { return unwrapProperty( this._accessibleRoleDescription ); }

  /**
   * Set the focus highlight for this Node. By default, the focus highlight will be a pink rectangle that
   * surrounds the Node's local bounds.  If focus highlight is set to 'invisible', the Node will not have
   * any highlighting when it receives focus.
   *
   * Use the local coordinate frame when drawing a custom highlight for this Node.
   */
  public setFocusHighlight( focusHighlight: Highlight ): void {
    if ( this._focusHighlight !== focusHighlight ) {
      this._focusHighlight = focusHighlight;

      // if the focus highlight is layerable in the scene graph, update visibility so that it is only
      // visible when associated Node has focus
      if ( this._focusHighlightLayerable ) {

        // if focus highlight is layerable, it must be a Node in the scene graph
        assert && assert( focusHighlight instanceof ParallelDOM ); // eslint-disable-line phet/no-simple-type-checking-assertions

        // the highlight starts off invisible, HighlightOverlay will make it visible when this Node has DOM focus
        ( focusHighlight as Node ).visible = false;
      }

      this.focusHighlightChangedEmitter.emit();
    }
  }

  public set focusHighlight( focusHighlight: Highlight ) { this.setFocusHighlight( focusHighlight ); }

  public get focusHighlight(): Highlight { return this.getFocusHighlight(); }

  /**
   * Get the focus highlight for this Node.
   */
  public getFocusHighlight(): Highlight {
    return this._focusHighlight;
  }

  /**
   * Setting a flag to break default and allow the focus highlight to be (z) layered into the scene graph.
   * This will set the visibility of the layered focus highlight, it will always be invisible until this Node has
   * focus.
   */
  public setFocusHighlightLayerable( focusHighlightLayerable: boolean ): void {

    if ( this._focusHighlightLayerable !== focusHighlightLayerable ) {
      this._focusHighlightLayerable = focusHighlightLayerable;

      // if a focus highlight is defined (it must be a Node), update its visibility so it is linked to focus
      // of the associated Node
      if ( this._focusHighlight ) {
        assert && assert( this._focusHighlight instanceof ParallelDOM );
        ( this._focusHighlight as Node ).visible = false;

        // emit that the highlight has changed and we may need to update its visual representation
        this.focusHighlightChangedEmitter.emit();
      }
    }
  }

  public set focusHighlightLayerable( focusHighlightLayerable: boolean ) { this.setFocusHighlightLayerable( focusHighlightLayerable ); }

  public get focusHighlightLayerable(): boolean { return this.getFocusHighlightLayerable(); }

  /**
   * Get the flag for if this Node is layerable in the scene graph (or if it is always on top, like the default).
   */
  public getFocusHighlightLayerable(): boolean {
    return this._focusHighlightLayerable;
  }

  /**
   * Set whether or not this Node has a group focus highlight. If this Node has a group focus highlight, an extra
   * focus highlight will surround this Node whenever a descendant Node has focus. Generally
   * useful to indicate nested keyboard navigation. If true, the group focus highlight will surround
   * this Node's local bounds. Otherwise, the Node will be used.
   *
   * TODO: Support more than one group focus highlight (multiple ancestors could have groupFocusHighlight), see https://github.com/phetsims/scenery/issues/1608
   */
  public setGroupFocusHighlight( groupHighlight: Node | boolean ): void {
    this._groupFocusHighlight = groupHighlight;
  }

  public set groupFocusHighlight( groupHighlight: Node | boolean ) { this.setGroupFocusHighlight( groupHighlight ); }

  public get groupFocusHighlight(): Node | boolean { return this.getGroupFocusHighlight(); }

  /**
   * Get whether or not this Node has a 'group' focus highlight, see setter for more information.
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

    const beforeOnly: Association[] = []; // Will hold all Nodes that will be removed.
    const afterOnly: Association[] = []; // Will hold all Nodes that will be "new" children (added)
    const inBoth: Association[] = []; // Child Nodes that "stay". Will be ordered for the "after" case.

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
   * Add an aria-labelledby association to this Node. The data in the associationObject will be implemented like
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

    // Flag that this Node is is being labelled by the other Node, so that if the other Node changes it can tell
    // this Node to restore the association appropriately.
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

    // remove the reference from the other Node back to this Node because we don't need it anymore
    removedObject[ 0 ].otherNode.removeNodeThatIsAriaLabelledByThisNode( this as unknown as Node );

    this.updateAriaLabelledbyAssociationsInPeers();
  }

  /**
   * Remove the reference to the Node that is using this Node's ID as an aria-labelledby value (scenery-internal)
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

    // if any other Nodes are aria-labelledby this Node, update those associations too. Since this Node's
    // pdom content needs to be recreated, they need to update their aria-labelledby associations accordingly.
    for ( let i = 0; i < this._nodesThatAreAriaLabelledbyThisNode.length; i++ ) {
      const otherNode = this._nodesThatAreAriaLabelledbyThisNode[ i ];
      otherNode.updateAriaLabelledbyAssociationsInPeers();
    }
  }

  /**
   * The list of Nodes that are aria-labelledby this Node (other Node's peer element will have this Node's Peer element's
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

    const beforeOnly: Association[] = []; // Will hold all Nodes that will be removed.
    const afterOnly: Association[] = []; // Will hold all Nodes that will be "new" children (added)
    const inBoth: Association[] = []; // Child Nodes that "stay". Will be ordered for the "after" case.
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
   * Add an aria-describedby association to this Node. The data in the associationObject will be implemented like
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

    // Flag that this Node is is being described by the other Node, so that if the other Node changes it can tell
    // this Node to restore the association appropriately.
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

    // remove the reference from the other Node back to this Node because we don't need it anymore
    removedObject[ 0 ].otherNode.removeNodeThatIsAriaDescribedByThisNode( this as unknown as Node );

    this.updateAriaDescribedbyAssociationsInPeers();
  }

  /**
   * Remove the reference to the Node that is using this Node's ID as an aria-describedby value (scenery-internal)
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

    // if any other Nodes are aria-describedby this Node, update those associations too. Since this Node's
    // pdom content needs to be recreated, they need to update their aria-describedby associations accordingly.
    // TODO: only use unique elements of the array (_.unique) https://github.com/phetsims/scenery/issues/1581
    for ( let i = 0; i < this._nodesThatAreAriaDescribedbyThisNode.length; i++ ) {
      const otherNode = this._nodesThatAreAriaDescribedbyThisNode[ i ];
      otherNode.updateAriaDescribedbyAssociationsInPeers();
    }
  }

  /**
   * The list of Nodes that are aria-describedby this Node (other Node's peer element will have this Node's Peer element's
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

    const beforeOnly: Association[] = []; // Will hold all Nodes that will be removed.
    const afterOnly: Association[] = []; // Will hold all Nodes that will be "new" children (added)
    const inBoth: Association[] = []; // Child Nodes that "stay". Will be ordered for the "after" case.
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
   * Add an aria-activeDescendant association to this Node. The data in the associationObject will be implemented like
   * "a peer's HTMLElement of this Node (specified with the string constant stored in `thisElementName`) will have an
   * aria-activeDescendant attribute with a value that includes the `otherNode`'s peer HTMLElement's id (specified with
   * `otherElementName`)."
   */
  public addActiveDescendantAssociation( associationObject: Association ): void {

    // TODO: assert if this associationObject is already in the association objects list! https://github.com/phetsims/scenery/issues/832
    this._activeDescendantAssociations.push( associationObject ); // Keep track of this association.

    // Flag that this Node is is being described by the other Node, so that if the other Node changes it can tell
    // this Node to restore the association appropriately.
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

    // remove the reference from the other Node back to this Node because we don't need it anymore
    removedObject[ 0 ].otherNode.removeNodeThatIsActiveDescendantThisNode( this as unknown as Node );

    this.updateActiveDescendantAssociationsInPeers();
  }

  /**
   * Remove the reference to the Node that is using this Node's ID as an aria-activeDescendant value (scenery-internal)
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

    // if any other Nodes are aria-activeDescendant this Node, update those associations too. Since this Node's
    // pdom content needs to be recreated, they need to update their aria-activeDescendant associations accordingly.
    // TODO: only use unique elements of the array (_.unique) https://github.com/phetsims/scenery/issues/1581
    for ( let i = 0; i < this._nodesThatAreActiveDescendantToThisNode.length; i++ ) {
      const otherNode = this._nodesThatAreActiveDescendantToThisNode[ i ];
      otherNode.updateActiveDescendantAssociationsInPeers();
    }
  }

  /**
   * The list of Nodes that are aria-activeDescendant this Node (other Node's peer element will have this Node's Peer element's
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
   * (first children first, last children last), determined by the children array. A Node must be connected to a scene
   * graph (via children) in order for pdomOrder to apply. Thus, `setPDOMOrder` cannot be used in exchange for
   * setting a Node as a child.
   *
   * In the general case, when pdomOrder is specified, it's an array of Nodes, with optionally one
   * element being a placeholder for "the rest of the children", signified by null. This means that, for
   * accessibility, it will act as if the children for this Node WERE the pdomOrder (potentially
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
   * - Nodes must be attached to a Display (in a scene graph) to be shown in a pdom order.
   * - You can't specify a Node in more than one pdomOrder, and you can't specify duplicates of a value
   *   in a pdomOrder.
   * - You can't specify an ancestor of a Node in that Node's pdomOrder
   *   (e.g. this.pdomOrder = this.parents ).
   *
   * Note that specifying something in a pdomOrder will effectively remove it from all of its parents for
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
    if ( assert && pdomOrder ) {
      pdomOrder.forEach( ( node, index ) => {
        assert && assert( node === null || node instanceof ParallelDOM,
          `Elements of pdomOrder should be either a Node or null. Element at index ${index} is: ${node}` );
      } );
      assert && assert( ( this as unknown as Node ).getTrails( node => _.includes( pdomOrder, node ) ).length === 0, 'pdomOrder should not include any ancestors or the Node itself' );
      assert && assert( pdomOrder.length === _.uniq( pdomOrder ).length, 'pdomOrder does not allow duplicate Nodes' );
    }

    // First a comparison to see if the order is switching to or from null
    let changed = ( this._pdomOrder === null && pdomOrder !== null ) ||
                  ( this._pdomOrder !== null && pdomOrder === null );

    if ( !changed && pdomOrder && this._pdomOrder ) {

      // We are comparing two arrays, so need to check contents for differences.
      changed = pdomOrder.length !== this._pdomOrder.length;

      if ( !changed ) {

        // Lengths are the same, so we need to look for content or order differences.
        for ( let i = 0; i < pdomOrder.length; i++ ) {
          if ( pdomOrder[ i ] !== this._pdomOrder[ i ] ) {
            changed = true;
            break;
          }
        }
      }
    }

    // Only update if it has changed
    if ( changed ) {
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
   * Returns the pdom (focus) order for this Node.
   *
   * Making changes to the returned array will not affect this Node's order. It returns a defensive copy.
   */
  public getPDOMOrder(): ( Node | null )[] | null {
    if ( this._pdomOrder ) {
      return this._pdomOrder.slice( 0 ); // create a defensive copy
    }
    return this._pdomOrder;
  }

  /**
   * Returns whether this Node has a pdomOrder that is effectively different than the default.
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
   * Returns our "PDOM parent" if available: the Node that specifies this Node in its pdomOrder.
   */
  public getPDOMParent(): Node | null {
    return this._pdomParent;
  }

  public get pdomParent(): Node | null { return this.getPDOMParent(); }

  /**
   * Returns the "effective" pdom children for the Node (which may be different based on the order or other
   * excluded subtrees).
   *
   * If there is no pdomOrder specified, this is basically "all children that don't have pdom parents"
   * (a Node has a "PDOM parent" if it is specified in a pdomOrder).
   *
   * Otherwise (if it has a pdomOrder), it is the pdomOrder, with the above list of Nodes placed
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

        // @ts-expect-error - TODO: best way to type? https://github.com/phetsims/scenery/issues/1581
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
   * Called when our pdomVisible Property changes values.
   */
  private onPdomVisiblePropertyChange( visible: boolean ): void {
    this._pdomDisplaysInfo.onPDOMVisibilityChange( visible );
  }

  /**
   * Sets what Property our pdomVisibleProperty is backed by, so that changes to this provided Property will change this
   * Node's pdom visibility, and vice versa. This does not change this._pdomVisibleProperty. See TinyForwardingProperty.setTargetProperty()
   * for more info.
   */
  public setPdomVisibleProperty( newTarget: TReadOnlyProperty<boolean> | null ): this {
    this._pdomVisibleProperty.setTargetProperty( newTarget );

    return this;
  }

  /**
   * See setPdomVisibleProperty() for more information
   */
  public set pdomVisibleProperty( property: TReadOnlyProperty<boolean> | null ) {
    this.setPdomVisibleProperty( property );
  }

  /**
   * See getPdomVisibleProperty() for more information
   */
  public get pdomVisibleProperty(): TProperty<boolean> {
    return this.getPdomVisibleProperty();
  }


  /**
   * Get this Node's pdomVisibleProperty. See Node.getVisibleProperty for more information
   */
  public getPdomVisibleProperty(): TProperty<boolean> {
    return this._pdomVisibleProperty;
  }

  /**
   * Hide completely from a screen reader and the browser by setting the hidden attribute on the Node's
   * representative DOM element. If the sibling DOM Elements have a container parent, the container
   * should be hidden so that all PDOM elements are hidden as well.  Hiding the element will remove it from the focus
   * order.
   */
  public setPDOMVisible( visible: boolean ): void {
    this.pdomVisibleProperty.value = visible;
  }

  public set pdomVisible( visible: boolean ) { this.setPDOMVisible( visible ); }

  public get pdomVisible(): boolean { return this.isPDOMVisible(); }

  /**
   * Get whether or not this Node's representative DOM element is visible.
   */
  public isPDOMVisible(): boolean {
    return this.pdomVisibleProperty.value;
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

  private invalidatePeerInputValue(): void {
    for ( let i = 0; i < this.pdomInstances.length; i++ ) {
      const peer = this.pdomInstances[ i ].peer!;
      peer.onInputValueChange();
    }
  }

  /**
   * Set the value of an input element.  Element must be a form element to support the value attribute. The input
   * value is converted to string since input values are generally string for HTML.
   */
  public setInputValue( inputValue: PDOMValueType | number | null ): void {
    assert && this._tagName && assert( _.includes( FORM_ELEMENTS, this._tagName.toUpperCase() ), 'dom element must be a form element to support value' );

    if ( inputValue !== this._inputValue ) {
      if ( isTReadOnlyProperty( this._inputValue ) && !this._inputValue.isDisposed ) {
        this._inputValue.unlink( this._onPDOMContentChangeListener );
      }

      this._inputValue = inputValue;

      if ( isTReadOnlyProperty( inputValue ) ) {
        inputValue.lazyLink( this._onPDOMContentChangeListener );
      }

      this.invalidatePeerInputValue();
    }
  }

  public set inputValue( value: PDOMValueType | number | null ) { this.setInputValue( value ); }

  public get inputValue(): string | number | null { return this.getInputValue(); }

  /**
   * Get the value of the element. Element must be a form element to support the value attribute.
   */
  public getInputValue(): string | null {
    let value: string | number | null;
    if ( isTReadOnlyProperty( this._inputValue ) ) {
      value = this._inputValue.value;
    }
    else {
      value = this._inputValue;
    }
    return value === null ? null : '' + value;
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
        type: 'property'
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

  /**
   * Sets all of the attributes for this Node's accessible content at once. See setPDOMAttribute for more information.
   *
   * Clears the old list of attributes before setting to this attribute list.
   */
  public setPDOMAttributes( attributes: PDOMAttribute[] ): void {

    // Remove all previous attributes.
    this.removePDOMAttributes();

    // Add the new attributes.
    for ( let i = 0; i < attributes.length; i++ ) {
      const attribute = attributes[ i ];
      this.setPDOMAttribute( attribute.attribute, attribute.value, attribute.options );
    }
  }

  public get pdomAttributes(): PDOMAttribute[] { return this.getPDOMAttributes(); }

  public set pdomAttributes( attributes: PDOMAttribute[] ) { this.setPDOMAttributes( attributes ); }

  /**
   * Set a particular attribute or property for this Node's primary sibling, generally to provide extra semantic information for
   * a screen reader.
   *
   * @param attribute - string naming the attribute
   * @param value - the value for the attribute, if boolean, then it will be set as a javascript property on the HTMLElement rather than an attribute
   * @param [providedOptions]
   */
  public setPDOMAttribute( attribute: string, value: Exclude<PDOMValueType, null> | boolean | number, providedOptions?: SetPDOMAttributeOptions ): void {

    assert && providedOptions && assert( Object.getPrototypeOf( providedOptions ) === Object.prototype,
      'Extra prototype on pdomAttribute options object is a code smell' );

    const options = optionize<SetPDOMAttributeOptions>()( {

      // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
      // for setting certain attributes (e.g. MathML).
      namespace: null,

      // set the "attribute" as a javascript property on the DOMElement instead of a DOM element attribute
      type: 'attribute',

      elementName: PEER_PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
    }, providedOptions );

    assert && assert( !ASSOCIATION_ATTRIBUTES.includes( attribute ), 'setPDOMAttribute does not support association attributes' );
    assert && options.namespace && assert( options.type === 'attribute', 'property-setting is not supported for custom namespaces' );

    // if the pdom attribute already exists in the list, remove it - no need
    // to remove from the peers, existing attributes will simply be replaced in the DOM
    for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
      const currentAttribute = this._pdomAttributes[ i ];
      if ( currentAttribute.attribute === attribute &&
           currentAttribute.options?.namespace === options.namespace &&
           currentAttribute.options?.elementName === options.elementName ) {

        // We can simplify the new value set as long as there isn't cleanup (from a Property listener) or logic change (from a different type)
        if ( !isTReadOnlyProperty( currentAttribute.value ) && currentAttribute.options.type === options.type ) {
          this._pdomAttributes.splice( i, 1 );
        }
        else {

          // Swapping type strategies should remove the attribute, so it can be set as a property/attribute correctly.
          this.removePDOMAttribute( currentAttribute.attribute, currentAttribute.options );
        }
      }
    }

    let listener: ( ( rawValue: string | boolean | number ) => void ) | null = ( rawValue: string | boolean | number ) => {
      assert && typeof rawValue === 'string' && validate( rawValue, Validation.STRING_WITHOUT_TEMPLATE_VARS_VALIDATOR );

      for ( let j = 0; j < this._pdomInstances.length; j++ ) {
        const peer = this._pdomInstances[ j ].peer!;
        peer.setAttributeToElement( attribute, rawValue, options );
      }
    };

    if ( isTReadOnlyProperty( value ) ) {
      // should run it once initially
      value.link( listener );
    }
    else {
      // Run it once and toss it, so we don't need to store the reference or unlink it later.
      // The listener ensures that the value is non-null.
      listener( value );
      listener = null;
    }

    this._pdomAttributes.push( {
      attribute: attribute,
      value: value,
      listener: listener,
      options: options
    } );

  }

  /**
   * Remove a particular attribute, removing the associated semantic information from the DOM element.
   *
   * It is HIGHLY recommended that you never call this function from an attribute set with `type:'property'`, see
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

      elementName: PEER_PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
    }, providedOptions );

    let attributeRemoved = false;
    for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
      if ( this._pdomAttributes[ i ].attribute === attribute &&
           this._pdomAttributes[ i ].options?.namespace === options.namespace &&
           this._pdomAttributes[ i ].options?.elementName === options.elementName ) {

        const oldAttribute = this._pdomAttributes[ i ];
        if ( oldAttribute.listener && isTReadOnlyProperty( oldAttribute.value ) && !oldAttribute.value.isDisposed ) {
          oldAttribute.value.unlink( oldAttribute.listener );
        }

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
   * Remove all attributes from this Node's dom element.
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

      elementName: PEER_PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
    }, providedOptions );

    let attributeFound = false;
    for ( let i = 0; i < this._pdomAttributes.length; i++ ) {
      if ( this._pdomAttributes[ i ].attribute === attribute &&
           this._pdomAttributes[ i ].options?.namespace === options.namespace &&
           this._pdomAttributes[ i ].options?.elementName === options.elementName ) {
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
      elementName: PEER_PRIMARY_SIBLING
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
      elementName: PEER_PRIMARY_SIBLING // see PDOMPeer.getElementName() for valid values, default to the primary sibling
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

        // after the override is set, update the focusability of the peer based on this Node's value for focusable
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
   * Get whether or not the Node is focusable. Use the focusOverride, and then default to browser defined
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
   * Node are observed so that the primary sibling is positioned correctly in the global coordinate frame.
   *
   * The transformSourceNode cannot use DAG for now because we need a unique trail to observe transforms.
   *
   * By default, transforms along trails to all of this Node's PDOMInstances are observed. But this
   * function can be used if you have a visual Node represented in the PDOM by a different Node in the scene
   * graph but still need the other Node's PDOM content positioned over the visual Node. For example, this could
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
   * Used by the animatedPanZoomSingleton. It will try to keep these bounds visible in the viewport when this Node
   * (or any ancestor) has a transform change while focused. This is useful if the bounds of your focusable
   * Node do not accurately surround the conceptual interactive component. If null, this Node's local bounds
   * are used.
   *
   * At this time, the Property cannot be changed after it is set.
   */
  public setFocusPanTargetBoundsProperty( boundsProperty: null | TReadOnlyProperty<Bounds2> ): void {

    // We may call this more than once with mutate
    if ( boundsProperty !== this._focusPanTargetBoundsProperty ) {
      assert && assert( !this._focusPanTargetBoundsProperty, 'Cannot change focusPanTargetBoundsProperty after it is set.' );
      this._focusPanTargetBoundsProperty = boundsProperty;
    }
  }

  /**
   * Returns the function for creating global bounds to keep in the viewport while the component has focus, see the
   * setFocusPanTargetBoundsProperty function for more information.
   */
  public getFocusPanTargetBoundsProperty(): null | TReadOnlyProperty<Bounds2> {
    return this._focusPanTargetBoundsProperty;
  }

  /**
   * See setFocusPanTargetBoundsProperty for more information.
   */
  public set focusPanTargetBoundsProperty( boundsProperty: null | TReadOnlyProperty<Bounds2> ) {
    this.setFocusPanTargetBoundsProperty( boundsProperty );
  }

  /**
   * See getFocusPanTargetBoundsProperty for more information.
   */
  public get focusPanTargetBoundsProperty(): null | TReadOnlyProperty<Bounds2> {
    return this.getFocusPanTargetBoundsProperty();
  }

  /**
   * Sets the direction that the global AnimatedPanZoomListener will pan while interacting with this Node. Pan will ONLY
   * occur in this dimension. This is especially useful for panning to large Nodes where panning to the center of the
   * Node would move other Nodes out of the viewport.
   *
   * Set to null for default behavior (panning in all directions).
   */
  public setLimitPanDirection( limitPanDirection: LimitPanDirection | null ): void {
    this._limitPanDirection = limitPanDirection;
  }

  /**
   * See setLimitPanDirection for more information.
   */
  public getLimitPanDirection(): LimitPanDirection | null {
    return this._limitPanDirection;
  }

  /**
   * See setLimitPanDirection for more information.
   * @param limitPanDirection
   */
  public set limitPanDirection( limitPanDirection: LimitPanDirection ) {
    this.setLimitPanDirection( limitPanDirection );
  }

  /**
   * See getLimitPanDirection for more information.
   */
  public get limitPanDirection(): LimitPanDirection | null {
    return this.getLimitPanDirection();
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
    this._excludeLabelSiblingFromInput = true;
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
  public addAccessibleResponse( utterance: TAlertable ): void {

    // Nothing to do if there is no content.
    if ( utterance === null ) {
      return;
    }

    // No description should be alerted if setting PhET-iO state, see https://github.com/phetsims/scenery/issues/1397
    if ( isSettingPhetioStateProperty.value ) {
      return;
    }

    // No description should be alerted if an archetype of a PhET-iO dynamic element, see https://github.com/phetsims/joist/issues/817
    if ( Tandem.PHET_IO_ENABLED && this.isInsidePhetioArchetype() ) {
      return;
    }

    const connectedDisplays = ( this as unknown as Node ).getConnectedDisplays();

    // Don't use `forEachUtterance` to prevent creating a closure for each usage of this function
    for ( let i = 0; i < connectedDisplays.length; i++ ) {
      const display = connectedDisplays[ i ];
      if ( display.isAccessible() ) {
        display.descriptionUtteranceQueue.addToBack( utterance );
      }
    }
  }

  /**
   * Helper method to add an accessible response of a specific category, with control over interruption behavior.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  private addCategorizedResponse( alertable: AlertableNoUtterance, alertBehavior: 'queue' | 'interrupt' ): void {
    if ( alertBehavior === 'interrupt' ) {
      this.forEachUtteranceQueue( queue => queue.clear() );
    }

    this.addAccessibleResponse( alertable );
  }

  /**
   * Add an object description to the utterance queue for screen readers. Object descriptions describe what an
   * object is or its current state.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  public addAccessibleObjectResponse( alertable: AlertableNoUtterance, alertBehavior: 'queue' | 'interrupt' = 'interrupt' ): void {
    this.addCategorizedResponse( alertable, alertBehavior );
  }

  /**
   * Add a context description to the utterance queue for screen readers. Context descriptions provide
   * information about the surrounding environment or the relationship of objects to one another.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  public addAccessibleContextResponse( alertable: AlertableNoUtterance, alertBehavior: 'queue' | 'interrupt' = 'interrupt' ): void {
    this.addCategorizedResponse( alertable, alertBehavior );
  }

  /**
   * Add a hint description to the utterance queue for screen readers. Hints provide guidance about
   * what the user can do or how they can interact with an object.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  public addAccessibleHelpResponse( alertable: AlertableNoUtterance, alertBehavior: 'queue' | 'interrupt' = 'interrupt' ): void {
    this.addCategorizedResponse( alertable, alertBehavior );
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
   * on this Node.
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
    let pruneStack: Node[] = []; // A list of Nodes to prune

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
      // count the number of times our Node appears in the pruneStack
      _.each( pruneStack, pruneNode => {
        if ( node === pruneNode ) {
          pruneCount++;
        }
      } );

      // If overridePruning is set, we ignore one reference to our Node in the prune stack. If there are two copies,
      // however, it means a Node was specified in a pdomOrder that already needs to be pruned (so we skip it instead
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

      // push specific focused Nodes to the stack
      pruneStack = pruneStack.concat( arrayPDOMOrder as Node[] );

      // Visiting trails to ordered Nodes.
      // @ts-expect-error
      _.each( arrayPDOMOrder, ( descendant: Node ) => {
        // Find all descendant references to the Node.
        // NOTE: We are not reordering trails (due to descendant constraints) if there is more than one instance for
        // this descendant Node.
        _.each( node.getLeafTrailsTo( descendant ), descendantTrail => {
          descendantTrail.removeAncestor(); // strip off 'node', so that we handle only children

          // same as the normal order, but adding a full trail (since we may be referencing a descendant Node)
          currentTrail.addDescendantTrail( descendantTrail );
          addTrailsForNode( descendant, true ); // 'true' overrides one reference in the prune stack (added above)
          currentTrail.removeDescendantTrail( descendantTrail );
        } );
      } );

      // Visit everything. If there is a pdomOrder, those trails were already visited, and will be excluded.
      const numChildren = node._children.length;
      for ( let i = 0; i < numChildren; i++ ) {
        const child = node._children[ i ];

        currentTrail.addDescendant( child, i );
        addTrailsForNode( child, false );
        currentTrail.removeDescendant();
      }

      // pop focused Nodes from the stack (that were added above)
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

    ( this as unknown as Node ).rendererSummaryRefreshEmitter.emit();
  }

  /**
   * Returns whether or not this Node has any representation for the Parallel DOM.
   * Note this is still true if the content is pdomVisible=false or is otherwise hidden.
   */
  public get hasPDOMContent(): boolean {
    return !!this._tagName || !!this._accessibleParagraph || !!this._accessibleHeading;
  }

  /**
   * Called when the Node is added as a child to this Node AND the Node's subtree contains pdom content.
   * We need to notify all Displays that can see this change, so that they can update the PDOMInstance tree.
   */
  protected onPDOMAddChild( node: Node ): void {
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( `onPDOMAddChild n#${node.id} (parent:n#${( this as unknown as Node ).id})` );
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

    // Find descendants with pdomOrders and check them against all of their ancestors/self
    assert && ( function recur( descendant ) {
      // Prune the search (because milliseconds don't grow on trees, even if we do have assertions enabled)
      if ( descendant._rendererSummary.hasNoPDOM() ) { return; }

      descendant.pdomOrder && assert( descendant.getTrails( node => _.includes( descendant.pdomOrder, node ) ).length === 0, 'pdomOrder should not include any ancestors or the Node itself' );
    } )( node );

    assert && PDOMTree.auditNodeForPDOMCycles( this as unknown as Node );

    this._pdomDisplaysInfo.onAddChild( node );

    PDOMTree.addChild( this as unknown as Node, node );

    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
  }

  /**
   * Called when the Node is removed as a child from this Node AND the Node's subtree contains pdom content.
   * We need to notify all Displays that can see this change, so that they can update the PDOMInstance tree.
   */
  protected onPDOMRemoveChild( node: Node ): void {
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.ParallelDOM( `onPDOMRemoveChild n#${node.id} (parent:n#${( this as unknown as Node ).id})` );
    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.push();

    this._pdomDisplaysInfo.onRemoveChild( node );

    PDOMTree.removeChild( this as unknown as Node, node );

    // make sure that the associations for aria-labelledby and aria-describedby are updated for Nodes associated
    // to this Node (they are pointing to this Node's IDs). https://github.com/phetsims/scenery/issues/816
    node.updateOtherNodesAriaLabelledby();
    node.updateOtherNodesAriaDescribedby();
    node.updateOtherNodesActiveDescendant();

    sceneryLog && sceneryLog.ParallelDOM && sceneryLog.pop();
  }

  /**
   * Called when this Node's children are reordered (with nothing added/removed).
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
  public updateLinkedElementForProperty<T>( tandemName: string, oldProperty?: TReadOnlyProperty<T> | null, newProperty?: TReadOnlyProperty<T> | null ): void {
    assert && assert( oldProperty !== newProperty, 'should not be called on same values' );

    // Only update linked elements if this Node is instrumented for PhET-iO
    if ( this.isPhetioInstrumented() ) {

      oldProperty && oldProperty instanceof ReadOnlyProperty && oldProperty.isPhetioInstrumented() && oldProperty instanceof PhetioObject && this.removeLinkedElements( oldProperty );

      const tandem = this.tandem.createTandem( tandemName );

      if ( newProperty && newProperty instanceof ReadOnlyProperty && newProperty.isPhetioInstrumented() && newProperty instanceof PhetioObject && tandem !== newProperty.tandem ) {
        this.addLinkedElement( newProperty, { tandemName: tandemName } );
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
   * Adds a PDOMInstance reference to our array. (scenery-internal)
   */
  public addPDOMInstance( pdomInstance: PDOMInstance ): void {
    this._pdomInstances.push( pdomInstance );
  }

  /**
   * Removes a PDOMInstance reference from our array. (scenery-internal)
   */
  public removePDOMInstance( pdomInstance: PDOMInstance ): void {
    const index = _.indexOf( this._pdomInstances, pdomInstance );
    assert && assert( index !== -1, 'Cannot remove a PDOMInstance from a Node if it was not there' );
    this._pdomInstances.splice( index, 1 );
  }

  public static BASIC_ACCESSIBLE_NAME_BEHAVIOR( node: Node, options: ParallelDOMOptions, accessibleName: PDOMValueType ): ParallelDOMOptions {
    if ( node.labelTagName && PDOMUtils.tagNameSupportsContent( node.labelTagName ) ) {
      options.labelContent = accessibleName;
    }
    else if ( node.tagName === 'input' ) {
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

  /**
   * The basic accessibleParagraph behavior function - sets the accessibleParagraphContent so that there is a paragraph
   * describing this Node.
   */
  public static BASIC_ACCESSIBLE_PARAGRAPH_BEHAVIOR( node: Node, options: ParallelDOMOptions, accessibleParagrah: PDOMValueType ): ParallelDOMOptions {
    options.accessibleParagraphContent = accessibleParagrah;
    return options;
  }

  /**
   * A heading behavior that forwards the accessibleName to the accessibleHeading. For components where
   * the accessibleName is logically represented with a heading, this allows you to still use the accessibleName
   * API.
   *
   * The accessibleName is cleared so there isn't a duplication.
   */
  public static HEADING_ACCESSIBLE_NAME_BEHAVIOR( node: Node, options: ParallelDOMOptions, accessibleName: PDOMValueType ): ParallelDOMOptions {
    options.accessibleName = null;
    options.accessibleHeading = accessibleName;
    return options;
  }

  /**
   * A behavior function for accessible name so that when accessibleName is set on the provided Node, it will be forwarded
   * to otherNode. This is useful when a component is composed of other Nodes that implement the accessibility,
   * but the high level API should be available for the entire component.
   */
  public static forwardAccessibleName( node: ParallelDOM, otherNode: ParallelDOM ): void {
    ParallelDOM.useDefaultTagName( node );
    node.accessibleNameBehavior = ( node: Node, options: ParallelDOMOptions, accessibleName: PDOMValueType, callbacksForOtherNodes: ( () => void )[] ) => {
      callbacksForOtherNodes.push( () => {
        otherNode.accessibleName = accessibleName;
      } );
      return options;
    };
  }

  /**
   * A behavior function for help text so that when accessibleHelpText is set on the provided 'node', it will be forwarded `otherNode`.
   * This is useful when a component is composed of other Nodes that implement the accessibility, but the high level API
   * should be available for the entire component.
   */
  public static forwardHelpText( node: ParallelDOM, otherNode: ParallelDOM ): void {
    ParallelDOM.useDefaultTagName( node );
    node.accessibleHelpTextBehavior = ( node: Node, options: ParallelDOMOptions, accessibleHelpText: PDOMValueType, callbacksForOtherNodes: ( () => void )[] ) => {
      callbacksForOtherNodes.push( () => {
        otherNode.accessibleHelpText = accessibleHelpText;
      } );
      return options;
    };
  }

  public static HELP_TEXT_BEFORE_CONTENT( node: Node, options: ParallelDOMOptions, accessibleHelpText: PDOMValueType ): ParallelDOMOptions {
    options.descriptionTagName = PDOMUtils.DEFAULT_DESCRIPTION_TAG_NAME;
    options.descriptionContent = accessibleHelpText;
    options.appendDescription = false;
    return options;
  }

  public static HELP_TEXT_AFTER_CONTENT( node: Node, options: ParallelDOMOptions, accessibleHelpText: PDOMValueType ): ParallelDOMOptions {
    options.descriptionTagName = PDOMUtils.DEFAULT_DESCRIPTION_TAG_NAME;
    options.descriptionContent = accessibleHelpText;
    options.appendDescription = true;
    return options;
  }

  /**
   * If the Node does not have a tagName yet, set it to the default.
   */
  private static useDefaultTagName( node: ParallelDOM ): void {
    if ( !node.tagName ) {
      node.tagName = DEFAULT_TAG_NAME;
    }
  }
}

scenery.register( 'ParallelDOM', ParallelDOM );
export { ACCESSIBILITY_OPTION_KEYS };